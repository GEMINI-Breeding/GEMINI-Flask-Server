import os
import sys
import re
import json
import math
import time
import shutil
import tempfile
import threading
import traceback
import gc
import pandas as pd
import numpy as np
import yaml
import torch
from PIL import Image, ImageFile
import rasterio
from rasterio.crs import CRS
from pyproj import Transformer
from rasterio.transform import Affine
from rasterio.warp import reproject, calculate_default_transform, Resampling
from rasterio.merge import merge

try:
    import gc
    import psutil
except ImportError:
    print("Warning: gc or psutil not available. Memory monitoring disabled.")
    gc = None
    psutil = None

# AgRowStitch imports
AGROWSTITCH_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../AgRowStitch"))
sys.path.append(AGROWSTITCH_PATH)

try:
    from panorama_maker.AgRowStitch import run as run_agrowstitch
except ImportError:
    print("Warning: AgRowStitch module not found. Some functions may not work.")
    run_agrowstitch = None

def pick_utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32700 if lat < 0 else 32600) + zone

def check_drone_orthomosaic_crs(drone_tiff_path: str):
    """
    Check the CRS and bounds of a drone orthomosaic to help with alignment debugging.
    """
    try:
        with rasterio.open(drone_tiff_path) as src:
            print(f"Drone orthomosaic CRS: {src.crs}")
            print(f"Drone orthomosaic bounds: {src.bounds}")
            print(f"Drone orthomosaic transform: {src.transform}")
            print(f"Drone orthomosaic dimensions: {src.width} x {src.height}")
            
            # Calculate drone GSD
            drone_gsd_x = abs(src.transform.a)  # pixel width in CRS units
            drone_gsd_y = abs(src.transform.e)  # pixel height in CRS units
            print(f"Drone orthomosaic GSD: x={drone_gsd_x:.6f} m/px, y={drone_gsd_y:.6f} m/px")
            
            # Convert bounds to WGS84 for comparison
            if src.crs != CRS.from_epsg(4326):
                transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                min_x, min_y, max_x, max_y = src.bounds
                min_lon, min_lat = transformer.transform(min_x, min_y)
                max_lon, max_lat = transformer.transform(max_x, max_y)
                print(f"Drone orthomosaic WGS84 bounds: ({min_lon:.6f}, {min_lat:.6f}) to ({max_lon:.6f}, {max_lat:.6f})")
            
            return {
                'crs': str(src.crs),
                'bounds': src.bounds,
                'transform': src.transform,
                'width': src.width,
                'height': src.height,
                'gsd_x': drone_gsd_x,
                'gsd_y': drone_gsd_y
            }
    except Exception as e:
        print(f"Error reading drone orthomosaic: {e}")
        return None

def fit_angle_pca(x: np.ndarray, y: np.ndarray) -> float:
    pts = np.column_stack([x, y]) - np.column_stack([x, y]).mean(axis=0)
    _, _, v = np.linalg.svd(pts, full_matrices=False)
    vx, vy = v[0]
    return math.atan2(vy, vx)

def compute_axes_extents(xs: np.ndarray, ys: np.ndarray, theta: float, buffer_frac: float = 0.05):
    """Project (xs, ys) to heading axis (u) and orthogonal axis (v);
    return buffered extents and widths in meters.
    
    Returns extents relative to the GPS track center (for proper coordinate transformation).
    """
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    u_vec = np.array([cos_t, sin_t])        # along-track
    v_vec = np.array([-sin_t, cos_t])       # cross-track

    # Center the GPS coordinates first
    gps_center_x = xs.mean()
    gps_center_y = ys.mean()
    xs_centered = xs - gps_center_x
    ys_centered = ys - gps_center_y
    
    pts_centered = np.column_stack([xs_centered, ys_centered])
    u_proj = pts_centered @ u_vec
    v_proj = pts_centered @ v_vec

    u_min, u_max = u_proj.min(), u_proj.max()
    v_min, v_max = v_proj.min(), v_proj.max()

    u_len = u_max - u_min
    v_len = v_max - v_min

    u_min -= u_len * buffer_frac * 0.5
    u_max += u_len * buffer_frac * 0.5
    v_min -= v_len * buffer_frac * 0.5
    v_max += v_len * buffer_frac * 0.5

    width_m  = u_max - u_min
    height_m = v_max - v_min
    return u_min, u_max, v_min, v_max, width_m, height_m


def build_rotated_affine(u_min, v_max, width_m, height_m, theta, px, py, gps_center_x, gps_center_y):
    """Return rasterio Affine with rotation encoded.
    
    Args:
        u_min, v_max: Extents in centered rotated coordinate system
        px, py: Pixel sizes in rotated coordinate system (width_m_axis/w, height_m_axis/h)
        gps_center_x, gps_center_y: Center of GPS track in UTM coordinates
    """
    theta_corrected = theta + math.pi
    cos_t, sin_t = math.cos(theta_corrected), math.sin(theta_corrected)

    # top-left in centered (u,v) coordinate system
    u_tl = u_min
    v_tl = v_max

    # Transform back to UTM coordinates:
    # 1. Apply rotation to get offset from center
    x_offset = u_tl * cos_t + v_tl * (-sin_t)
    y_offset = u_tl * sin_t + v_tl * ( cos_t)
    
    # 2. Add GPS center to get absolute UTM coordinates
    x_tl = gps_center_x + x_offset
    y_tl = gps_center_y + y_offset

    # The affine transform maps image pixels to world coordinates
    # px, py are already in the rotated (u,v) coordinate system
    
    a = px * cos_t       # dX for +1 col
    d = px * sin_t       # dY for +1 col
    b = py * sin_t       # dX for +1 row  = -(-py)*sin, simplified
    e = -py * cos_t      # dY for +1 row
    
    print(f"DEBUG AFFINE: px={px:.6f}, py={py:.6f}, theta={math.degrees(theta):.2f}°")
    print(f"DEBUG AFFINE: cos_t={cos_t:.6f}, sin_t={sin_t:.6f}")
    print(f"DEBUG AFFINE: a={a:.6f}, b={b:.6f}, d={d:.6f}, e={e:.6f}")
    
    c = x_tl
    f = y_tl
    return Affine(a, b, c, d, e, f)

def calc_gsd_from_intrinsics(camera_intrinsics: dict,
                             camera_height_m: float,
                             img_w: int,
                             img_h: int):
    """
    Very simple GSD estimate (m/px) from intrinsics + height.
    Assumes vertical imaging, small tilt.
    Returns average of fx, fy based estimate.
    """
    try:
        if 'cameraData' in camera_intrinsics:
            # OAK-D style
            cam = None
            for c in camera_intrinsics['cameraData']:
                if 'intrinsicMatrix' in c and 'width' in c and 'height' in c:
                    # pick the one matching the stitch images (or just largest)
                    if c['width'] == img_w or c['height'] == img_h:
                        cam = c
                        break
            if cam is None:
                cam = camera_intrinsics['cameraData'][0]
            K = cam['intrinsicMatrix']
            fx, fy = K[0], K[4]
        else:
            # legacy
            K = camera_intrinsics.get('camera_matrix', [[1000,0,960],[0,1000,540],[0,0,1]])
            fx, fy = K[0][0], K[1][1]

        # f(px) -> f(m): GSD ≈ H / f(px)   (ignoring sensor pixel pitch, HFOV used elsewhere)
        gsd_x = camera_height_m / fx
        gsd_y = camera_height_m / fy
        return (gsd_x + gsd_y) / 2.0
    except Exception:
        return None
    
def determine_stitch_direction_from_gps(xs, ys, theta_deg, rover_direction):
    """
    Determine stitching direction from GPS trajectory analysis.
    """
    # Analyze the primary movement direction
    dx = xs[-1] - xs[0]
    dy = ys[-1] - ys[0]
    
    # Determine if movement is more horizontal or vertical
    if abs(dx) > abs(dy):
        # Horizontal movement
        direction = "RIGHT" if dx > 0 else "LEFT"
    else:
        # Vertical movement  
        direction = "UP" if dy > 0 else "DOWN"
    
    print(f"  GPS-derived stitching direction: {direction} (dx={dx:.3f}m, dy={dy:.3f}m)")
    return direction

def estimate_cross_track_auto(xs, ys, theta, h_pixels, px_along, strategy,
                              intr_gsd=None, fixed_min=None, ratio=None, k_sigma=5,
                              stitch_direction=None, img_w=None, img_h=None):
    """
    Returns (height_m_axis, v_min, v_max) based on chosen strategy.
    - strategy: 'intrinsics', 'std', 'ratio', 'fixed', 'hybrid'
    - stitch_direction: Direction of stitching (LEFT, RIGHT, UP, DOWN)
    - img_w, img_h: Image dimensions for direction-aware hybrid calculation
    
    Returns v_min, v_max in centered coordinate system.
    """
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    v_vec = np.array([-sin_t, cos_t])
    
    # Center the GPS coordinates first (same as compute_axes_extents)
    gps_center_x = xs.mean()
    gps_center_y = ys.mean()
    xs_centered = xs - gps_center_x
    ys_centered = ys - gps_center_y
    
    v_vals = np.column_stack([xs_centered, ys_centered]) @ v_vec
    v_min = v_vals.min()
    v_max = v_vals.max()
    height_gps = v_max - v_min

    if strategy == 'intrinsics' and intr_gsd is not None:
        height_need = intr_gsd * h_pixels
    elif strategy == 'hybrid' and intr_gsd is not None:
        # HYBRID: Direction-aware approach
        if stitch_direction in ['LEFT', 'RIGHT']:
            # Horizontal stitching: GPS = along-track, intrinsics height = cross-track
            intrinsics_height = intr_gsd * img_h if img_h else intr_gsd * h_pixels
            height_need = max(height_gps, intrinsics_height)
            print(f"  Hybrid (H-stitch): GPS={height_gps:.3f}m, intrinsics_height={intrinsics_height:.3f}m, using={height_need:.3f}m")
        elif stitch_direction in ['UP', 'DOWN']:
            # Vertical stitching: GPS = along-track, intrinsics width = cross-track
            intrinsics_width = intr_gsd * img_w if img_w else intr_gsd * h_pixels
            height_need = max(height_gps, intrinsics_width)
            print(f"  Hybrid (V-stitch): GPS={height_gps:.3f}m, intrinsics_width={intrinsics_width:.3f}m, using={height_need:.3f}m")
        else:
            # Unknown direction: use default hybrid approach
            intrinsics_height = intr_gsd * h_pixels
            height_need = max(height_gps, intrinsics_height)
            print(f"  Hybrid (unknown dir): GPS={height_gps:.3f}m, intrinsics={intrinsics_height:.3f}m, using={height_need:.3f}m")
    elif strategy == 'std':
        sigma = np.std(v_vals)
        height_need = max(height_gps, k_sigma * sigma)
    elif strategy == 'ratio' and ratio is not None:
        # ratio is (h/w)
        width_m_axis = px_along * (len(xs) * 0 + 1)  # px_along already width_m / w
        height_need = max(height_gps, width_m_axis * ratio)
    elif strategy == 'fixed' and fixed_min is not None:
        height_need = max(height_gps, fixed_min)
    else:
        height_need = height_gps  # fallback

    # Center on original v_center
    if height_need > height_gps:
        print(f"  Centering: {height_need:.3f}m > {height_gps:.3f}m")
        v_center = 0.5 * (v_min + v_max)
        v_min = v_center - height_need / 2.0
        v_max = v_center + height_need / 2.0
        height_gps = height_need

    return height_gps, v_min, v_max

def reproject_to_wgs84(src_path: str, dst_path: str):
    """
    Reproject a UTM GeoTIFF to WGS84 (EPSG:4326).
    """
    print(f"Reprojecting {src_path} → {dst_path} (EPSG:4326)")
    try:
        with rasterio.open(src_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, "EPSG:4326", src.width, src.height, *src.bounds
            )
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': CRS.from_epsg(4326),
                'transform': transform,
                'width': width,
                'height': height
            })
            
            with rasterio.open(dst_path, "w", **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs="EPSG:4326",
                        resampling=Resampling.bilinear
                    )
        print(f"Finished reprojection to {dst_path}")
        return True
    except Exception as e:
        print(f"Error during reprojection: {str(e)}")
        return False

def combine_utm_tiffs_to_mosaic(versioned_output_path: str, processed_plots: list):
    """
    Combine individual UTM GeoTIFFs into one mosaic with proper handling of rotated plots.
    Each plot may have its own rotation, so we need to reproject them to a common grid.
    """
    print("=== STARTING MOSAIC CREATION WITH ROTATION HANDLING ===")
    
    # 1) build list of existing UTM files
    utm_files = []
    for pid in processed_plots:
        p = os.path.join(versioned_output_path, f"georeferenced_plot_{pid}_utm.tif")
        if os.path.exists(p):
            utm_files.append(p)
            print(f"Found UTM file for plot {pid}: {p}")
        else:
            print(f"Warning: missing {p}")
    
    if not utm_files:
        print("No UTM files found; skipping mosaic.")
        return False

    print(f"Processing {len(utm_files)} UTM files for mosaic creation")

    # 2) if only one, just copy
    combined_utm = os.path.join(versioned_output_path, "combined_mosaic_utm.tif")
    if len(utm_files) == 1:
        print("Only one plot found, copying directly")
        shutil.copy2(utm_files[0], combined_utm)
    else:
        try:
            print("Multiple plots found, creating mosaic with rotation handling...")
            
            # Open all source files and analyze their properties
            srcs = []
            transforms = []
            bounds_list = []
            
            for fp in utm_files:
                src = rasterio.open(fp)
                srcs.append(src)
                transforms.append(src.transform)
                bounds_list.append(src.bounds)
                
                # Check if the raster has rotation
                has_rotation = src.transform.b != 0 or src.transform.d != 0
                print(f"File: {os.path.basename(fp)}")
                print(f"  Transform: {src.transform}")
                print(f"  Has rotation: {has_rotation}")
                print(f"  Bounds: {src.bounds}")
                print(f"  Size: {src.width}x{src.height}")
            
            # Calculate overall bounds
            min_x = min(bounds[0] for bounds in bounds_list)
            min_y = min(bounds[1] for bounds in bounds_list)
            max_x = max(bounds[2] for bounds in bounds_list)
            max_y = max(bounds[3] for bounds in bounds_list)
            
            print(f"Overall bounds: ({min_x}, {min_y}, {max_x}, {max_y})")
            
            # Determine output resolution based on the first source
            # Use the minimum pixel size from all sources to maintain detail
            pixel_sizes = []
            for i, src in enumerate(srcs):
                # For rotated transforms, calculate the actual pixel size
                # Handle case where transform components might be very small or malformed
                pixel_size_x = abs(src.transform.a) if abs(src.transform.a) > 1e-10 else None
                pixel_size_y = abs(src.transform.e) if abs(src.transform.e) > 1e-10 else None
                
                # If transform seems malformed, calculate from bounds and dimensions
                if pixel_size_x is None or pixel_size_y is None:
                    bounds = src.bounds
                    calc_pixel_size_x = (bounds.right - bounds.left) / src.width
                    calc_pixel_size_y = (bounds.top - bounds.bottom) / src.height
                    print("  Transform appears malformed, using bounds-based calculation:")
                    print(f"    Bounds-based pixel size: x={calc_pixel_size_x:.6f}, y={calc_pixel_size_y:.6f}")
                    pixel_size_x = calc_pixel_size_x
                    pixel_size_y = calc_pixel_size_y
                
                # Take the smaller dimension to preserve detail
                effective_pixel_size = min(pixel_size_x, pixel_size_y)
                pixel_sizes.append(effective_pixel_size)
                
                print(f"  Plot {processed_plots[i]} effective pixel size: {effective_pixel_size:.6f} meters/pixel")
            
            output_pixel_size = min(pixel_sizes)
            print(f"Using output pixel size: {output_pixel_size:.6f} meters/pixel")
            
            # Calculate output dimensions with safety checks
            out_width = int((max_x - min_x) / output_pixel_size)
            out_height = int((max_y - min_y) / output_pixel_size)
            
            # Safety check: prevent extremely large mosaics that could cause memory issues
            max_dimension = 5000  # Maximum dimension in pixels
            if out_width > max_dimension or out_height > max_dimension:
                print(f"WARNING: Calculated output size {out_width}x{out_height} is too large!")
                
                # Calculate a safer pixel size
                safe_pixel_size_x = (max_x - min_x) / max_dimension
                safe_pixel_size_y = (max_y - min_y) / max_dimension
                output_pixel_size = max(safe_pixel_size_x, safe_pixel_size_y, output_pixel_size)
                
                out_width = int((max_x - min_x) / output_pixel_size)
                out_height = int((max_y - min_y) / output_pixel_size)
                
                print(f"Adjusted to safer pixel size: {output_pixel_size:.6f} meters/pixel")
                print(f"New output size: {out_width}x{out_height} pixels")
            
            print(f"Output mosaic size: {out_width}x{out_height} pixels")
            
            # Create output transform (no rotation - aligned to coordinate system)
            out_transform = rasterio.transform.from_bounds(
                min_x, min_y, max_x, max_y, out_width, out_height
            )
            
            print(f"Output transform: {out_transform}")
            
            # Create output array with white background (255 for all RGB channels)
            mosaic = np.full((3, out_height, out_width), 255, dtype=np.uint8)
            
            # Reproject each source to the common grid
            for i, src in enumerate(srcs):
                print(f"Reprojecting plot {processed_plots[i]}...")
                
                # Create temporary arrays for reprojection with white background
                temp_data = np.full((3, out_height, out_width), 255, dtype=np.uint8)
                
                # Reproject each band
                for band_idx in range(3):
                    reproject(
                        source=rasterio.band(src, band_idx + 1),
                        destination=temp_data[band_idx],
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=out_transform,
                        dst_crs=src.crs,  # Same CRS, just different grid
                        resampling=Resampling.bilinear,
                        dst_nodata=0
                    )
                
                # Combine with mosaic using "first" method (non-zero values take precedence)
                for band_idx in range(3):
                    mask = temp_data[band_idx] != 0
                    mosaic[band_idx][mask] = temp_data[band_idx][mask]
                
                print(f"Completed reprojection for plot {processed_plots[i]}")
            
            # Write the mosaic
            out_meta = srcs[0].meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_height,
                "width": out_width,
                "transform": out_transform,
                "crs": srcs[0].crs,
                "compress": "lzw",
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
                "count": 3,
                "dtype": "uint8"
            })

            with rasterio.open(combined_utm, "w", **out_meta) as dst:
                dst.write(mosaic)
                print(f"Written mosaic with {3} bands")

            # Close all source files
            for src in srcs:
                src.close()
            
            print("Successfully created UTM mosaic with proper rotation handling")
            
        except Exception as e:
            print(f"Error creating mosaic: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            
            # Close any open files
            for src in srcs:
                try:
                    src.close()
                except Exception:
                    pass
            return False

    print(f"Written UTM mosaic → {combined_utm}")

    # 4) reproject to WGS84
    try:
        print("Creating WGS84 version of mosaic...")
        combined_wgs84 = os.path.join(versioned_output_path, "combined_mosaic.tif")
        
        with rasterio.open(combined_utm) as src:
            dst_crs = CRS.from_epsg(4326)
            transform, w, h = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
            meta = src.meta.copy()
            meta.update({
                "crs": dst_crs,
                "transform": transform,
                "width": w,
                "height": h
            })
            with rasterio.open(combined_wgs84, "w", **meta) as dst:
                for i in range(1, src.count+1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear
                    )

        print(f"Written WGS84 mosaic → {combined_wgs84}")
        print("=== MOSAIC CREATION COMPLETED SUCCESSFULLY ===")
        return True
        
    except Exception as e:
        print(f"Error creating WGS84 mosaic: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

def georeference_plot(plot_index,
                      plot_data,
                      plot_stitched_path,
                      versioned_output_path,
                      has_stitch_direction,
                      camera_intrinsics=None,
                      gps_offset_x=0.0,
                      gps_offset_y=0.0):
    """
    Georeference a stitched plot image using GPS points at image centers.

    Args:
        gps_offset_x: Manual GPS offset in meters (UTM X direction)
        gps_offset_y: Manual GPS offset in meters (UTM Y direction)

    Modes:
      - GPS_ONLY:          only GPS extents, enforce a reasonable min cross-track (stats/ratio/fixed)
      - INTRINSICS_BOUND:  enforce min cross-track using intrinsics-derived GSD

    Keeps all your prints.
    """
    # ----------- CONFIG SWITCHES -----------
    MODE = "HYBRID"                # "GPS_ONLY", "INTRINSICS_BOUND", or "HYBRID" - HYBRID recommended for stitched images
    camera_height = 1.2            # meters (hard-coded)
    min_cross_track_aspect = 0.0   # floor = 30% of ideal (width * h/w) if GPS is too thin
    
    # GPS validation and correction options
    AUTO_CORRECT_GPS_LENGTH = True  # Automatically correct GPS length if it seems unreasonable
    GPS_LENGTH_TOLERANCE = 0.3      # Allow 30% difference between PCA and direct GPS distance
    PREFERRED_LENGTH_METHOD = "direct"  # "direct", "pca", or "path" - which GPS length to prefer

    # Strategy options for helper
    CROSS_TRACK_STRATEGY = "fixed"
    FIXED_MIN_CT = 1.0   # used if strategy == 'fixed'
    K_SIGMA = 5          # used if strategy == 'std'

    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        Image.MAX_IMAGE_PIXELS = None

        base_name = "AgRowStitch"
        print(f"\n========== Georeferencing Plot {plot_index} ==========")

        # ---- find mosaic ----
        mosaic_candidates = [
            f"{base_name}_plot-id-{plot_index}.png",
            f"{base_name}_plot-id-{plot_index}.tif",
            f"full_res_mosaic_temp_plot_{plot_index}.png",
            f"full_res_mosaic_temp_plot_{plot_index}.tif",
        ]
        src_file = None
        for fname in mosaic_candidates:
            p = os.path.join(versioned_output_path, fname)
            if os.path.exists(p):
                src_file = p
                break
        if src_file is None:
            print(f"[Plot {plot_index}] No mosaic file found in {versioned_output_path}")
            return False
        print(f"[Plot {plot_index}] Using mosaic file: {src_file}")

        # ---- GPS sanity ----
        if "lat" not in plot_data.columns or "lon" not in plot_data.columns:
            print(f"[Plot {plot_index}] lat/lon columns missing.")
            return False

        plot_df = (plot_data.sort_values("/top/rgb_file")
                   if "/top/rgb_file" in plot_data.columns
                   else plot_data.sort_index())

        lats = plot_df["lat"].dropna().to_numpy()
        lons = plot_df["lon"].dropna().to_numpy()
        if lats.size == 0 or lons.size == 0:
            print(f"[Plot {plot_index}] No valid lat/lon values.")
            return False

        first_lat, first_lon = lats[0], lons[0]
        last_lat,  last_lon  = lats[-1], lons[-1]
        print(f"[Plot {plot_index}] First GPS  ({first_lat:.6f}, {first_lon:.6f})")
        print(f"[Plot {plot_index}] Last  GPS  ({last_lat:.6f}, {last_lon:.6f})")

        lat_diff = last_lat - first_lat
        lon_diff = last_lon - first_lon
        if abs(lat_diff) > abs(lon_diff):
            rover_direction = "NORTH" if lat_diff > 0 else "SOUTH"
        else:
            rover_direction = "EAST" if lon_diff > 0 else "WEST"
        print(f"[Plot {plot_index}] Rover direction guess: {rover_direction}")

        if has_stitch_direction and "stitch_direction" in plot_df.columns:
            stitch_direction = str(plot_df["stitch_direction"].iloc[0]).upper()
        else:
            stitch_direction = "UNKNOWN"
        print(f"[Plot {plot_index}] Stitching direction (column): {stitch_direction}")

        # ---- read mosaic ----
        with Image.open(src_file) as im:
            if im.mode != "RGB":
                im = im.convert("RGB")
            img_array = np.array(im)
        h, w = img_array.shape[:2]
        print(f"[Plot {plot_index}] Image size: {w} x {h}")

        # ---- choose UTM ----
        center_lat = (lats.min() + lats.max()) / 2.0
        center_lon = (lons.min() + lons.max()) / 2.0
        utm_epsg = pick_utm_epsg(center_lon, center_lat)
        print(f"[Plot {plot_index}] Using UTM CRS: EPSG:{utm_epsg}")
        print(f"[Plot {plot_index}] Plot-specific center coordinates: lat={center_lat:.6f}, lon={center_lon:.6f}")

        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
        xs, ys = transformer.transform(lons, lats)
        
        # DIAGNOSTIC: Show actual GPS positioning for each plot
        print(f"[Plot {plot_index}] GPS POSITIONING DIAGNOSTICS:")
        print(f"  Plot GPS bounds in UTM: X({xs.min():.3f} to {xs.max():.3f}), Y({ys.min():.3f} to {ys.max():.3f})")
        print(f"  Plot center in UTM: X={xs.mean():.3f}m, Y={ys.mean():.3f}m")
        print(f"  GPS track extent: {xs.max()-xs.min():.3f}m x {ys.max()-ys.min():.3f}m")

        # Apply manual GPS offset if provided
        if gps_offset_x != 0.0 or gps_offset_y != 0.0:
            xs = xs + gps_offset_x
            ys = ys + gps_offset_y
            print(f"[Plot {plot_index}] Applied GPS offset: x={gps_offset_x:.3f}m, y={gps_offset_y:.3f}m")

        # If no stitch direction available, determine from GPS trajectory
        if stitch_direction == "UNKNOWN" or stitch_direction not in ['LEFT', 'RIGHT', 'UP', 'DOWN']:
            stitch_direction = determine_stitch_direction_from_gps(xs, ys, 0, rover_direction)  # theta_deg not available yet
            print(f"[Plot {plot_index}] Using GPS-derived stitching direction: {stitch_direction}")

        # ---- GPS accuracy assessment ----
        gps_distances = []
        for i in range(1, len(xs)):
            dist = np.sqrt((xs[i] - xs[i-1])**2 + (ys[i] - ys[i-1])**2)
            gps_distances.append(dist)
        
        if gps_distances:
            avg_gps_spacing = np.mean(gps_distances)
            std_gps_spacing = np.std(gps_distances)
            print(f"[Plot {plot_index}] GPS point spacing: avg={avg_gps_spacing:.3f}m, std={std_gps_spacing:.3f}m")
            print(f"[Plot {plot_index}] GPS extent: {xs.max()-xs.min():.3f}m x {ys.max()-ys.min():.3f}m")

        # ---- PCA heading ----
        theta = fit_angle_pca(xs, ys)
        theta_deg = math.degrees(theta) % 360
        print(f"[Plot {plot_index}] PCA trajectory angle: {theta_deg:.2f}° (0° = +X/East)")

        # Alternative angle calculation using first and last points
        dx = xs[-1] - xs[0]
        dy = ys[-1] - ys[0]
        simple_angle = math.atan2(dy, dx)
        simple_angle_deg = math.degrees(simple_angle) % 360
        angle_diff = abs(theta_deg - simple_angle_deg)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        print(f"[Plot {plot_index}] Simple angle (first->last): {simple_angle_deg:.2f}°, diff from PCA: {angle_diff:.2f}°")
        
        # Use simple angle if PCA and simple angle differ significantly (suggests noisy GPS)
        if angle_diff > 30:  # More than 30 degrees difference
            print(f"[Plot {plot_index}] Large angle difference detected, using simple first->last angle")
            theta = simple_angle
            theta_deg = simple_angle_deg

        # enforce left→right order (first should be left)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        u_first = cos_t * xs[0] + sin_t * ys[0]
        u_last  = cos_t * xs[-1] + sin_t * ys[-1]
        if u_last < u_first:
            theta += math.pi
            theta_deg = (theta_deg + 180) % 360
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            print(f"[Plot {plot_index}] Flipped θ by 180° to match mosaic L→R order")

        # ---- axis extents from GPS ----
        u_min, u_max, v_min, v_max, width_m_axis, height_m_axis = compute_axes_extents(
            xs, ys, theta, buffer_frac=0.0
        )
        print(f"[Plot {plot_index}] Raw axis extents: width={width_m_axis:.3f}m, height={height_m_axis:.3f}m")
        
        # Get GPS center for coordinate transformation
        gps_center_x = xs.mean()
        gps_center_y = ys.mean()
        
        # DIAGNOSTIC: Show actual positioning coordinates (now correctly centered)
        print(f"[Plot {plot_index}] POSITIONING DIAGNOSTICS:")
        print(f"  GPS center in UTM: X={gps_center_x:.3f}m, Y={gps_center_y:.3f}m")
        print(f"  Centered rotated coordinates: u_min={u_min:.3f}m, u_max={u_max:.3f}m")
        print(f"  Centered rotated coordinates: v_min={v_min:.3f}m, v_max={v_max:.3f}m")
        
        # Calculate where the top-left corner will be positioned in UTM
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        x_offset = u_min * cos_t + v_max * (-sin_t)
        y_offset = u_min * sin_t + v_max * cos_t
        utm_top_left_x = gps_center_x + x_offset
        utm_top_left_y = gps_center_y + y_offset
        print(f"  Plot will be positioned at UTM: ({utm_top_left_x:.3f}, {utm_top_left_y:.3f}) as top-left corner")
        
        # ---- Validate GPS extents against expected plot length ----
        # Calculate direct distance between first and last GPS points
        direct_gps_distance = np.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2)
        total_gps_path_length = sum(np.sqrt(np.diff(xs)**2 + np.diff(ys)**2))
        
        print(f"[Plot {plot_index}] GPS VALIDATION:")
        print(f"  Direct distance (first→last): {direct_gps_distance:.3f}m")
        print(f"  Total GPS path length: {total_gps_path_length:.3f}m")
        print(f"  PCA-derived width (along-track): {width_m_axis:.3f}m")
        print(f"  Ratio (PCA/direct): {width_m_axis/direct_gps_distance:.3f}")
        
        # Check for reasonable agreement between different length measures
        max_reasonable_ratio = 1.0 + GPS_LENGTH_TOLERANCE  # Allow tolerance difference
        min_reasonable_ratio = 1.0 - GPS_LENGTH_TOLERANCE  # Don't allow PCA to be too different
        
        length_correction_applied = False
        if AUTO_CORRECT_GPS_LENGTH:
            if width_m_axis > direct_gps_distance * max_reasonable_ratio:
                print(f"[Plot {plot_index}] WARNING: PCA width seems too large compared to direct GPS distance")
                print("  Auto-correcting using direct GPS distance")
                length_correction_applied = True
            elif width_m_axis < direct_gps_distance * min_reasonable_ratio:
                print(f"[Plot {plot_index}] WARNING: PCA width seems too small compared to direct GPS distance")
                print("  Auto-correcting using direct GPS distance")
                length_correction_applied = True
            elif PREFERRED_LENGTH_METHOD == "direct":
                print(f"[Plot {plot_index}] Using preferred method: direct GPS distance")
                length_correction_applied = True
            elif PREFERRED_LENGTH_METHOD == "path" and abs(total_gps_path_length - width_m_axis) < abs(direct_gps_distance - width_m_axis):
                print(f"[Plot {plot_index}] Using preferred method: total GPS path length")
                # Use path length instead of direct distance
                direct_gps_distance = total_gps_path_length
                length_correction_applied = True
        else:
            # Just warn but don't correct
            if width_m_axis > direct_gps_distance * max_reasonable_ratio:
                print(f"[Plot {plot_index}] WARNING: PCA width seems too large compared to direct GPS distance")
                print("  Consider using direct GPS distance as reference")
            elif width_m_axis < direct_gps_distance * min_reasonable_ratio:
                print(f"[Plot {plot_index}] WARNING: PCA width seems too small compared to direct GPS distance")
                print("  This might indicate GPS noise or poor trajectory alignment")
        
        # Apply length correction if needed
        if length_correction_applied:
            print(f"[Plot {plot_index}] Applying GPS length correction")
            # Recalculate extents centered on the trajectory but with corrected length
            # The u_min, u_max from compute_axes_extents are already centered, so we just
            # need to scale them based on the corrected width
            old_width = width_m_axis
            width_m_axis = direct_gps_distance
            
            # Scale the centered extents proportionally
            current_width = u_max - u_min
            scale_factor = width_m_axis / current_width
            u_center = (u_min + u_max) / 2.0
            u_half_width = (u_max - u_min) / 2.0 * scale_factor
            u_min = u_center - u_half_width
            u_max = u_center + u_half_width
            print(f"  Corrected width_m_axis: {old_width:.3f}m → {width_m_axis:.3f}m")

        # ---- cross-track decision using helper ----
        px_along = width_m_axis / w  # meters per pixel along-track

        intr_gsd = None
        if CROSS_TRACK_STRATEGY == "intrinsics" and camera_intrinsics:
            intr_gsd = calc_gsd_from_intrinsics(camera_intrinsics, camera_height, w, h)
            if intr_gsd is not None:
                print(f"[Plot {plot_index}] Intrinsics GSD: {intr_gsd:.6f} m/px")

        ideal_height_m = width_m_axis * (h / w)
        min_height_from_intr = (intr_gsd * h) if intr_gsd is not None else 0.0
        min_height_allowed = max(ideal_height_m * min_cross_track_aspect, min_height_from_intr)

        height_m_axis, v_min, v_max = estimate_cross_track_auto(
            xs, ys, theta,
            h_pixels=h,
            px_along=px_along,
            strategy=CROSS_TRACK_STRATEGY,
            intr_gsd=intr_gsd,
            fixed_min=FIXED_MIN_CT,
            ratio=(h / w),
            k_sigma=K_SIGMA,
            stitch_direction=stitch_direction,
            img_w=w,
            img_h=h
        )

        if height_m_axis < min_height_allowed:
            print(f"[Plot {plot_index}] Cross-track too small ({height_m_axis:.3f}m). Expanding to {min_height_allowed:.3f}m.")
            v_center = 0.5 * (v_min + v_max)
            v_min = v_center - min_height_allowed / 2.0
            v_max = v_center + min_height_allowed / 2.0
            height_m_axis = min_height_allowed

        # ---- pixel sizes ----
        # Calculate pixel sizes directly from the final calculated extents
        px = width_m_axis / w  # meters per pixel in X direction
        py = height_m_axis / h  # meters per pixel in Y direction
        
        print(f"[Plot {plot_index}] Using calculated extents for pixel scale")
        print(f"  Width: {width_m_axis:.3f}m ÷ {w}px = {px:.6f} m/px")
        print(f"  Height: {height_m_axis:.3f}m ÷ {h}px = {py:.6f} m/px")
        
        print(f"[Plot {plot_index}] SCALE DIAGNOSTICS:")
        print(f"  Image dimensions: {w} x {h} pixels")
        print(f"  Final extent: {width_m_axis:.3f}m x {height_m_axis:.3f}m")
        print(f"  Final pixel size: x={px:.6f} m/px, y={py:.6f} m/px")
        print(f"  Rotated bounds: u_min={u_min:.3f}m, u_max={u_max:.3f}m, v_min={v_min:.3f}m, v_max={v_max:.3f}m")
        print(f"  Calculated width check: u_max - u_min = {u_max - u_min:.3f}m (should match width_m_axis: {width_m_axis:.3f}m)")
        print(f"  Calculated height check: v_max - v_min = {v_max - v_min:.3f}m (should match height_m_axis: {height_m_axis:.3f}m)")
        
        # Verify extent consistency
        width_diff = abs((u_max - u_min) - width_m_axis)
        height_diff = abs((v_max - v_min) - height_m_axis)
        if width_diff > 0.001:  # tolerance of 1mm
            print(f"  WARNING: Width mismatch of {width_diff:.6f}m between bounds and final extent!")
        if height_diff > 0.001:  # tolerance of 1mm
            print(f"  WARNING: Height mismatch of {height_diff:.6f}m between bounds and final extent!")
        
        if width_diff <= 0.001 and height_diff <= 0.001:
            print("  Extent consistency verified - TIF will be positioned correctly within bounds")
        
        if intr_gsd is not None:
            print(f"  Single-frame intrinsics GSD: {intr_gsd:.6f} m/px (for comparison only)")
            # Calculate GPS-derived pixel size from original GPS extents
            gps_width = xs.max() - xs.min()
            gps_height = ys.max() - ys.min()
            gps_px_x = gps_width / w if w > 0 else 0
            gps_px_y = gps_height / h if h > 0 else 0
            print(f"  Original GPS extent: {gps_width:.3f}m x {gps_height:.3f}m")
            print(f"  GPS-derived pixel size: x={gps_px_x:.6f} m/px, y={gps_px_y:.6f} m/px")
            if MODE == "HYBRID":
                print(f"  Note: Using HYBRID approach - GPS for along-track, intrinsics for cross-track width (direction: {stitch_direction})")
            else:
                print("  Note: Using GPS-derived scale for stitched panorama (intrinsics are for single frames)")
        else:
            print("  Note: Using GPS-derived scale (recommended for stitched images)")

        gsd_avg = (px + py) / 2.0
        print(f"[Plot {plot_index}] Final ground coverage: {width_m_axis:.2f}m x {height_m_axis:.2f}m")
        print(f"[Plot {plot_index}] Pixel size / GSD: {gsd_avg:.6f} m/px (x: {px:.6f}, y: {py:.6f})")

        # ---- affine transform ----
        transform = build_rotated_affine(u_min, v_max, width_m_axis, height_m_axis, theta, px, py, gps_center_x, gps_center_y)

        print(f"[Plot {plot_index}] Affine transform:")
        print(f"    a={transform.a:.8f}, b={transform.b:.8f}, c={transform.c:.3f}")
        print(f"    d={transform.d:.8f}, e={transform.e:.8f}, f={transform.f:.3f}")
        print(f"[Plot {plot_index}] Transform positions plot at UTM: ({transform.c:.3f}, {transform.f:.3f})")
        print(f"[Plot {plot_index}] Rotation encoded: {theta_deg:.2f}°; rover guess: {rover_direction}")
        
        # Verify what dimensions this affine transform will actually produce
        # Calculate the four corners of the image in world coordinates
        corners_image = [(0, 0), (w, 0), (w, h), (0, h)]  # top-left, top-right, bottom-right, bottom-left
        corners_world = [transform * corner for corner in corners_image]
        
        # Extract X and Y coordinates
        xs_corners = [corner[0] for corner in corners_world]
        ys_corners = [corner[1] for corner in corners_world]
        
        # Calculate actual dimensions that will appear in QGIS
        actual_width = max(xs_corners) - min(xs_corners)
        actual_height = max(ys_corners) - min(ys_corners)
        
        print(f"[Plot {plot_index}] VERIFICATION - Actual TIF dimensions in QGIS:")
        print(f"    Expected: {width_m_axis:.3f}m x {height_m_axis:.3f}m")
        print(f"    Calculated: {actual_width:.3f}m x {actual_height:.3f}m")
        print(f"    Image corners in UTM: {corners_world}")
        
        if abs(actual_width - width_m_axis) > 0.1 or abs(actual_height - height_m_axis) > 0.1:
            print("    WARNING: Dimension mismatch detected!")
            print(f"    Width difference: {abs(actual_width - width_m_axis):.3f}m")
            print(f"    Height difference: {abs(actual_height - height_m_axis):.3f}m")

        # ---- write UTM GeoTIFF ----
        utm_out = os.path.join(versioned_output_path, f"georeferenced_plot_{plot_index}_utm.tif")
        print(f"[Plot {plot_index}] Writing UTM GeoTIFF: {utm_out}")
        with rasterio.open(
            utm_out, "w",
            driver="GTiff",
            height=h,
            width=w,
            count=3,
            dtype=img_array.dtype,
            crs=CRS.from_epsg(utm_epsg),
            transform=transform,
            compress="lzw",
            tiled=True,
            blockxsize=512,
            blockysize=512
        ) as dst:
            for i in range(3):
                dst.write(img_array[:, :, i], i + 1)
        print(f"[Plot {plot_index}] UTM GeoTIFF written.")

        print(f"[Plot {plot_index}] DONE. Outputs:")
        print(f"    UTM GeoTIFF : {utm_out}")
        # print(f"    WGS84 GeoTIFF: {wgs_out}")
        print("====================================================\n")
        return True

    except Exception as e:
        print(f"[Plot {plot_index}] ERROR during georeferencing: {e}")
        return False

def run_stitch_all_plots(msgs_synced_path, image_path, config_path, custom_options, save_path, calibration_path=None, stitch_stop_event=None, final_progress_callback=None):
    """
    Function to run stitching for all plots in background
    """
    try:
        # Load camera calibration if available
        camera_intrinsics = None
        if calibration_path and os.path.exists(calibration_path):
            try:
                with open(calibration_path, 'r') as f:
                    camera_intrinsics = json.load(f)
                print(f"Loaded camera calibration from: {calibration_path}")
            except Exception as e:
                print(f"Warning: Could not load camera calibration: {e}")
        
        # Read msgs_synced.csv to get plot information
        msgs_df = pd.read_csv(msgs_synced_path)
        print(f"Loaded msgs_synced.csv with {len(msgs_df)} rows")
        
        # Check required columns
        required_columns = ['plot_index', '/top/rgb_file']
        missing_columns = [col for col in required_columns if col not in msgs_df.columns]
        if missing_columns:
            print(f"Error: Missing columns in msgs_synced.csv: {missing_columns}")
            return
        
        # Get unique plot IDs
        unique_plots = msgs_df['plot_index'].unique()
        print(f"Found {len(unique_plots)} unique plots: {unique_plots}")

        # Check for stitch_direction column (optional)
        has_stitch_direction = 'stitch_direction' in msgs_df.columns
        if not has_stitch_direction:
            print("Warning: 'stitch_direction' column not found in msgs_synced.csv. Using default direction.")

        # Create a single versioned output directory for all outputs
        base_name = "AgRowStitch"
        version = 0
        existing_dirs = os.listdir(save_path) if os.path.exists(save_path) else []
        version_pattern = re.compile(re.escape(base_name) + r"_v(\d+)")
        
        for dir_name in existing_dirs:
            match = version_pattern.match(dir_name)
            if match:
                existing_version = int(match.group(1))
                version = max(version, existing_version + 1)
        
        # Create the main versioned output directory
        versioned_dir_name = f"{base_name}_v{version}"
        versioned_output_path = os.path.join(save_path, versioned_dir_name)
        os.makedirs(versioned_output_path, exist_ok=True)
        print(f"Created versioned output directory: {versioned_output_path}")

        # Process each plot
        processed_plots = []
        failed_plots = []
        
        print("\n========== STARTING PLOT PROCESSING ==========")
        print(f"Total plots to process: {len(unique_plots)}")
        print(f"Plot indices: {list(unique_plots)}")
        print(f"Stop event status: {stitch_stop_event.is_set() if stitch_stop_event else 'No stop event'}")
        print("===============================================")
        
        counter = 0
        for plot_index in unique_plots:
            if stitch_stop_event and stitch_stop_event.is_set():
                print("Stitching process stopped by user")
                break
            
            # Skip plot index 0
            if plot_index == -1:
                print(f"Skipping plot {plot_index} (plot index -1)")
                continue
                
            try:
                print(f"\n========== Processing plot {plot_index} ==========")
                print(f"Plot {plot_index}: Starting processing...")
                
                # Filter data for current plot
                plot_data = msgs_df[msgs_df['plot_index'] == plot_index]
                print(f"Plot {plot_index}: Found {len(plot_data)} data rows")

                # Get image filenames for this plot
                image_files = plot_data['/top/rgb_file'].dropna().tolist()
                if not image_files:
                    print(f"Plot {plot_index}: No images found, skipping")
                    failed_plots.append(plot_index)
                    continue
                
                print(f"Plot {plot_index}: Found {len(image_files)} images")
                
                # Create temporary directory for this plot's images
                plot_temp_dir = os.path.join(os.path.dirname(image_path), f"temp_plot_{plot_index}")
                if os.path.exists(plot_temp_dir):
                    shutil.rmtree(plot_temp_dir)
                os.makedirs(plot_temp_dir, exist_ok=True)
                
                # Copy images for this plot to temporary directory
                copied_files = []
                for img_file in image_files:
                    img_file = img_file.split('/')[-1]  # Get the filename only
                    src_path = os.path.join(image_path, img_file)
                    dst_path = os.path.join(plot_temp_dir, img_file)
                    if os.path.exists(src_path):
                        shutil.copy2(src_path, dst_path)
                        copied_files.append(img_file)
                    else:
                        print(f"Warning: Image {img_file} not found at {src_path}")
                
                if not copied_files:
                    print(f"No valid images copied for plot {plot_index}, skipping")
                    shutil.rmtree(plot_temp_dir)
                    failed_plots.append(plot_index)
                    continue
                
                print(f"Copied {len(copied_files)} images for plot {plot_index}")
                
                # Load original config
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)

                # Update fields for this plot
                config['image_directory'] = plot_temp_dir
                config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
                
                # Set stitching direction if available
                if has_stitch_direction:
                    stitching_direction = plot_data['stitch_direction'].iloc[0]
                    
                    # capitalize the stitching direction
                    if isinstance(stitching_direction, str):
                        stitching_direction = stitching_direction.strip().upper()
                        
                    print(f"Stitching direction for plot {plot_index}: {stitching_direction}")
                    if pd.notna(stitching_direction):
                        config['stitching_direction'] = stitching_direction
                        print(f"Set stitching_direction to {stitching_direction} for plot {plot_index}")
                else:
                    print(f"No stitching_direction found for plot {plot_index}, using default direction")

                # Apply any user-specified custom options
                if isinstance(custom_options, str):
                    custom_options = yaml.safe_load(custom_options)
                if custom_options and isinstance(custom_options, dict):
                    config.update(custom_options)

                # Save to temp config file
                with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix=f"_plot_{plot_index}.yaml") as tmpfile:
                    yaml.safe_dump(config, tmpfile)
                    temp_config_path = tmpfile.name

                cpu_count = os.cpu_count()
                
                # Create plot-specific output paths - use the shared final_mosaics directory
                plot_stitched_path = os.path.join(os.path.dirname(image_path), "final_mosaics")
                
                # Run stitching for this plot
                print(f"Starting stitching for plot {plot_index}")
                
                # Run the stitching process for this plot
                run_stitch_process_for_plot(temp_config_path, cpu_count, plot_stitched_path, versioned_output_path, plot_index, stitch_stop_event)
                
                # Georeference the individual plot at full resolution (maximum quality)
                print(f"Georeferencing individual plot {plot_index} at full resolution...")
                georeference_success = georeference_plot(plot_index, plot_data, plot_stitched_path, versioned_output_path, has_stitch_direction, camera_intrinsics)
                if georeference_success:
                    print(f"Successfully georeferenced plot {plot_index}")
                else:
                    print(f"Warning: Failed to georeference plot {plot_index}")
                
                # Clean up temporary files
                os.unlink(temp_config_path)
                shutil.rmtree(plot_temp_dir)
                
                processed_plots.append(plot_index)
                print(f"Successfully processed plot {plot_index}")
                
                # DEBUG: Break after specified number of plots for testing
                counter += 1
                # if counter >= 6:
                #     print(f"DEBUG: Breaking after {counter} plots to test combined mosaic creation")
                #     break
                
            except Exception as e:
                print(f"ERROR: Exception processing plot {plot_index}: {str(e)}")
                print(f"Exception type: {type(e).__name__}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                failed_plots.append(plot_index)
                
                # Clean up on error
                try:
                    if 'plot_temp_dir' in locals() and os.path.exists(plot_temp_dir):
                        shutil.rmtree(plot_temp_dir)
                        print(f"Cleaned up temp directory for plot {plot_index}")
                except Exception as cleanup_error:
                    print(f"Error cleaning up temp directory: {cleanup_error}")
                    
                try:
                    if 'temp_config_path' in locals() and os.path.exists(temp_config_path):
                        os.unlink(temp_config_path)
                        print(f"Cleaned up temp config for plot {plot_index}")
                except Exception as cleanup_error:
                    print(f"Error cleaning up temp config: {cleanup_error}")

        # Summary of processing
        print("\n========== PROCESSING SUMMARY ==========")
        print(f"Total plots to process: {len(unique_plots)}")
        print(f"Successfully processed: {len(processed_plots)}")
        print(f"Failed: {len(failed_plots)}")
        if processed_plots:
            print(f"Successful plots: {processed_plots}")
        if failed_plots:
            print(f"Failed plots: {failed_plots}")
        print("=========================================")
        
        result_message = f"Stitching completed. Processed plots: {processed_plots}"
        if failed_plots:
            result_message += f", Failed plots: {failed_plots}"
        
        print(result_message)
        
        # Individual plots completed - return results for separate mosaic creation
        if processed_plots:
            print(f"Successfully processed and georeferenced {len(processed_plots)} individual plots at full resolution")
            print("Individual georeferenced TIFF files created for maximum quality")
            print("Note: Combined mosaic will be created separately to avoid multiprocessing issues")
            # Don't set progress here - let the finally block in complete_stitch_workflow handle it
        else:
            print("No plots were successfully processed")
            
        # Return the results for external mosaic creation
        return {
            'processed_plots': processed_plots,
            'failed_plots': failed_plots,
            'versioned_output_path': versioned_output_path
        }
        
    except Exception as e:
        print(f"Error in run_stitch_all_plots: {str(e)}")


def create_combined_mosaic_separate(versioned_output_path: str, processed_plots: list, progress_callback=None, monitoring_stop_event=None):
    """
    Create combined mosaic separately from the main stitching process to avoid multiprocessing issues.
    This function is designed to be called after individual plot processing is complete.
    """
    try:
        print("=== STARTING SEPARATE MOSAIC CREATION ===")
        
        # Cancel monitoring thread since individual plot processing is done
        if monitoring_stop_event:
            print("Stopping monitoring thread - individual plot processing completed")
            monitoring_stop_event.set()
            time.sleep(1)  # Give monitoring thread a moment to stop cleanly
            print("Monitoring thread stopped")
        
        print(f"Creating combined mosaic from {len(processed_plots)} processed plots")
        print(f"Output directory: {versioned_output_path}")
        
        if not processed_plots:
            print("No processed plots provided for mosaic creation")
            if progress_callback:
                progress_callback(100)
            return False
        
        # Create combined mosaic from all UTM files
        print("Creating combined mosaic from individual UTM files...")
        mosaic_success = combine_utm_tiffs_to_mosaic(versioned_output_path, processed_plots)
        
        if mosaic_success:
            print("SUCCESS: Combined mosaic created successfully")
            print("Individual georeferenced TIFF files available for maximum quality")
            print("Combined mosaic also created for overview purposes")
        else:
            print("WARNING: Failed to create combined mosaic")
            print("Individual georeferenced TIFF files still available for maximum quality")
        if progress_callback:
            progress_callback(100)
        
        print("=== MOSAIC CREATION COMPLETED ===")
        return mosaic_success
        
    except Exception as e:
        print(f"ERROR in create_combined_mosaic_separate: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        if progress_callback:
            progress_callback(100)
        return False


def run_stitch_process_for_plot(config_path, cpu_count, stitched_path, versioned_output_path, plot_index, stitch_stop_event=None):
    """Run the stitching process for a single plot"""
    print(f"Stitching started for plot {plot_index} on thread {threading.current_thread().name}")

    try:
        # Check if AgRowStitch is available
        if run_agrowstitch is None:
            print(f"ERROR: AgRowStitch module not available for plot {plot_index}")
            raise ImportError("AgRowStitch module not found or could not be imported")
        
        # Run the AgRowStitch process
        print(f"Calling run_agrowstitch with config: {config_path} and cpu_count: {cpu_count}")
        agrowstitch_result = run_agrowstitch(config_path, cpu_count)
        print(f"AgRowStitch result type: {type(agrowstitch_result)}")
        print(f"AgRowStitch result: {agrowstitch_result}")
        
        # Check if the result is iterable
        if agrowstitch_result is None:
            print(f"AgRowStitch returned None for plot {plot_index}")
            # Don't return immediately, continue to check for output files
        else:
            # Try to iterate through the result
            try:
                if hasattr(agrowstitch_result, '__iter__') and not isinstance(agrowstitch_result, (str, bytes)):
                    for step in agrowstitch_result:
                        if stitch_stop_event and stitch_stop_event.is_set():
                            print(f"Stitching canceled by user for plot {plot_index}.")
                            return
                        print(f"Plot {plot_index} - Step: {step}")
                else:
                    print(f"AgRowStitch result is not iterable for plot {plot_index}, but process may have completed")
            except TypeError as e:
                print(f"AgRowStitch result is not iterable for plot {plot_index}: {e}")
                # If it's not iterable, assume the process completed successfully
                print(f"Assuming stitching completed successfully for plot {plot_index}")
            except Exception as e:
                print(f"Error during AgRowStitch iteration for plot {plot_index}: {e}")
                # Don't return, continue to check for output files
        
        print(f"AgRowStitch processing completed for plot {plot_index}, checking for output files...")

        # Now stitching is done — look for the actual generated files
        # Check if the stitched_path exists, if not wait a bit for it to be created
        wait_count = 0
        max_wait_attempts = 30  # Increased wait time
        while not os.path.exists(stitched_path) and wait_count < max_wait_attempts:
            print(f"Waiting for stitched output directory: {stitched_path} (attempt {wait_count + 1}/{max_wait_attempts})")
            time.sleep(5)  # Increased wait time per attempt
            wait_count += 1
        
        if not os.path.exists(stitched_path):
            print(f"Stitched output directory not found for plot {plot_index}: {stitched_path}")
            # Try to find alternative output directories
            possible_dirs = [
                os.path.join(os.path.dirname(stitched_path), "output"),
                os.path.join(os.path.dirname(stitched_path), "results"),
                os.path.join(os.path.dirname(stitched_path), "final_mosaics"),
                os.path.dirname(stitched_path)  # Check parent directory
            ]
            
            for alt_dir in possible_dirs:
                if os.path.exists(alt_dir):
                    print(f"Found alternative directory: {alt_dir}")
                    all_files = os.listdir(alt_dir)
                    print(f"Files in {alt_dir}: {all_files}")
                    
                    # Look for any files related to this plot
                    plot_files = [f for f in all_files if str(plot_index) in f or 'plot' in f.lower()]
                    if plot_files:
                        stitched_path = alt_dir
                        print(f"Using alternative directory {alt_dir} with files: {plot_files}")
                        break
            else:
                print(f"No suitable output directory found for plot {plot_index}")
                return
        
        # List all files in the stitched directory to see what was actually generated
        all_files = os.listdir(stitched_path)
        print(f"Files found in {stitched_path}: {all_files}")
        
        # Define image extensions we care about for the main stitched output
        image_extensions = ['.png', '.tif', '.tiff', '.jpg', '.jpeg']
        
        # Look for the main stitched image file - prioritize by likely patterns
        main_image_file = None
        
        # Priority 1: Look for files with specific patterns that contain the plot_index
        priority_patterns = [
            f"full_res_mosaic_temp_plot_{plot_index}",
            f"temp_plot_{plot_index}",
            f"mosaic_temp_plot_{plot_index}",
            f"plot_{plot_index}_mosaic",
            f"plot_{plot_index}"
        ]
        
        for pattern in priority_patterns:
            matching_files = [f for f in all_files 
                            if pattern in f and any(f.lower().endswith(ext) for ext in image_extensions)]
            if matching_files:
                # Take the largest file if multiple matches
                main_image_file = max(matching_files, key=lambda f: os.path.getsize(os.path.join(stitched_path, f)))
                print(f"Found main image using pattern '{pattern}': {main_image_file}")
                break
        
        # Priority 2: If no specific pattern found, look for any image files containing the plot_index
        if not main_image_file:
            plot_index_str = str(plot_index)
            candidate_files = [f for f in all_files 
                             if plot_index_str in f and any(f.lower().endswith(ext) for ext in image_extensions)]
            
            if candidate_files:
                # Prefer files with "mosaic" or "stitch" in the name
                mosaic_files = [f for f in candidate_files if 'mosaic' in f.lower() or 'stitch' in f.lower()]
                if mosaic_files:
                    main_image_file = max(mosaic_files, key=lambda f: os.path.getsize(os.path.join(stitched_path, f)))
                else:
                    # Take the largest image file containing plot_index
                    main_image_file = max(candidate_files, key=lambda f: os.path.getsize(os.path.join(stitched_path, f)))
                print(f"Found main image by plot_index matching: {main_image_file}")
        
        # Priority 3: If still no file found and there's only one image file, use it
        if not main_image_file:
            all_images = [f for f in all_files if any(f.lower().endswith(ext) for ext in image_extensions)]
            if len(all_images) == 1:
                main_image_file = all_images[0]
                print(f"Using single available image file: {main_image_file}")
            elif len(all_images) > 1:
                print(f"Multiple image files found but none match plot patterns: {all_images}")
                print(f"Cannot determine which is the main stitched output for plot {plot_index}")
                return
        
        if not main_image_file:
            print(f"No suitable stitched image file found for plot {plot_index}")
            return
        
        # Verify the main image file exists and has content
        src_file_path = os.path.join(stitched_path, main_image_file)
        if not os.path.exists(src_file_path):
            print(f"Main image file does not exist: {src_file_path}")
            return
            
        file_size = os.path.getsize(src_file_path)
        if file_size == 0:
            print(f"Main image file is empty: {src_file_path}")
            return
            
        print(f"Processing main stitched image: {main_image_file} (size: {file_size} bytes)")
        
        # Create the standardized filename that other parts of the system expect
        file_extension = os.path.splitext(main_image_file)[1]
        standardized_name = f"full_res_mosaic_temp_plot_{plot_index}{file_extension}"
        dst_file_path = os.path.join(versioned_output_path, standardized_name)
        
        # Copy the main stitched image with the standardized name
        try:
            shutil.copy2(src_file_path, dst_file_path)
            print(f"Successfully copied and renamed stitched image: {main_image_file} -> {standardized_name}")
            
            # Verify the copied file
            if os.path.exists(dst_file_path) and os.path.getsize(dst_file_path) > 0:
                print(f"Verified copied file: {standardized_name} ({os.path.getsize(dst_file_path)} bytes)")
            else:
                print(f"ERROR: Copied file verification failed for {standardized_name}")
                return
                
        except Exception as e:
            print(f"Error copying main stitched image: {str(e)}")
            return
        
        # Optional: Copy additional files (logs, metadata, etc.) but with clear naming
        # Only copy these if they exist and are useful
        additional_files_to_copy = []
        
        # Look for log files
        log_files = [f for f in all_files if f.endswith('.log') and str(plot_index) in f]
        if log_files:
            for log_file in log_files:
                additional_files_to_copy.append((log_file, f"AgRowStitch_plot-id-{plot_index}.log"))
        
        # Look for CSV files (but don't rename them to avoid confusion with the main image)
        csv_files = [f for f in all_files if f.endswith('.csv') and str(plot_index) in f]
        if csv_files:
            for csv_file in csv_files:
                # Keep original CSV name but prefix with AgRowStitch for clarity
                csv_name = f"AgRowStitch_plot-id-{plot_index}.csv"
                additional_files_to_copy.append((csv_file, csv_name))
        
        # Copy additional files if any
        additional_copied = 0
        for src_name, dst_name in additional_files_to_copy:
            try:
                src_path = os.path.join(stitched_path, src_name)
                dst_path = os.path.join(versioned_output_path, dst_name)
                
                if os.path.exists(src_path) and os.path.getsize(src_path) > 0:
                    shutil.copy2(src_path, dst_path)
                    print(f"Copied additional file: {src_name} -> {dst_name}")
                    additional_copied += 1
            except Exception as e:
                print(f"Warning: Could not copy additional file {src_name}: {str(e)}")
        
        print(f"Completed processing plot {plot_index}:")
        print(f"  - Main stitched image: {standardized_name}")
        print(f"  - Additional files copied: {additional_copied}")
        print(f"  - Output directory: {versioned_output_path}")
            
    except Exception as e:
        print(f"Error in stitching process for plot {plot_index}: {str(e)}")
        raise


def monitor_stitch_updates_multi_plot(final_mosaics_dir, processed_plots, latest_data_callback=None, stop_event=None):
    """Monitor multiple plot log files and aggregate progress"""
    try:
        # Wait for the final_mosaics directory to be created
        while not os.path.exists(final_mosaics_dir):
            print("Waiting for final_mosaics directory to be created...")
            time.sleep(5)
            # Check if we should stop early
            if stop_event and stop_event.is_set():
                print("Monitoring stopped early - stop event set")
                return

        print(f"Monitoring log files in: {final_mosaics_dir}")
        
        # Track completed plots
        completed_plots = set()
        total_plots = len(processed_plots)
        
        while True:
            # Check if we should stop
            if stop_event and stop_event.is_set():
                print("Monitoring stopped - stop event set")
                break
                
            # Check each plot's log file to see if it's completed
            for plot_index in processed_plots:
                if plot_index not in completed_plots:
                    log_file = os.path.join(final_mosaics_dir, f"temp_plot_{plot_index}.log")
                    
                    if os.path.exists(log_file):
                        # Check if this plot is done by looking for completion indicators
                        if is_plot_completed(log_file):
                            completed_plots.add(plot_index)
                            print(f"Plot {plot_index} completed!")
            
            # Calculate progress based on completed plots (cap at 95% to let main workflow handle 100%)
            if len(completed_plots) == total_plots:
                progress_percentage = 95  # Cap at 95% when all individual plots done
            else:
                progress_percentage = int((len(completed_plots) / total_plots) * 95)  # Scale to 95% max
            
            # Update progress using callback if provided
            if latest_data_callback:
                latest_data_callback(progress_percentage)
            
            # Check if all plots are complete
            if len(completed_plots) == total_plots:
                print("All individual plots completed and georeferenced!")
                print("Monitoring thread stopping - main workflow will handle final completion")
                # Don't set 100% here - let the main workflow handle final completion
                # Stop the monitoring thread cleanly so it doesn't interfere with final progress
                break
                
            time.sleep(3)  # Check every 3 seconds

    except Exception as e:
        print(f"Error in log monitoring: {e}")
    finally:
        print("Monitoring thread terminated - no more progress updates from this thread")


def is_plot_completed(log_file):
    """Check if a plot is completed by looking for completion indicators in the log"""
    try:
        with open(log_file, 'r') as file:
            content = file.read()
            
        # Look for completion indicators
        completion_indicators = [
            "Done",
            "Saving mosaic at full resolution",
            "Process completed",
            "Finished"
        ]
        
        for indicator in completion_indicators:
            if indicator in content:
                return True
                
        return False
        
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
        return False


def monitor_stitch_updates(progress_file, latest_data_callback=None):
    """Monitor single stitching progress file (legacy function)"""
    try:
        total_images = None  # we still extract total images once!

        final_steps = [
            "Retrieving batches",
            "Straightening batches",
            "Extracting batch features",
            "Matching batch features",
            "Stitching batches",
            "Straightening final mosaic",
            "Saving mosaic at full resolution",
            "Saving mosaic at resized resolution",
            "Deleting intermediate images",
            "Done"
        ]

        final_step_map = {step: idx for idx, step in enumerate(final_steps)}
        
        # Wait for the log file to be created
        while not os.path.exists(progress_file):
            print("Waiting for log file to be created...")
            time.sleep(5)  # Check every 5 seconds

        print("Log file found. Monitoring for updates.")

        with open(progress_file, 'r') as file:
            file.seek(0, os.SEEK_END)  # Start at end, tail new lines

            while True:
                line = file.readline()
                if not line:
                    time.sleep(1)
                    continue

                line = line.strip()

                # extract total images once
                if total_images is None:
                    match = re.search(r'Found (\d+) images', line)
                    if match:
                        total_images = int(match.group(1))
                        print(f"Total images detected: {total_images}")
                        continue

                # use actual match info to estimate progress up to 90%
                if "Succesfully matched image" in line and total_images:
                    match = re.search(r'Succesfully matched image (\d+) and (\d+)', line)
                    if match:
                        img2 = int(match.group(2))
                        percent = min(int((img2 / total_images) * 90), 90)
                        if latest_data_callback:
                            latest_data_callback(percent)

                # final steps take us from 90% to 100%
                for step in final_steps:
                    if step in line:
                        step_idx = final_step_map[step]
                        percent = 90 + int((step_idx / len(final_steps)) * 10)
                        if latest_data_callback:
                            latest_data_callback(percent)
                        
                        if step == "Done":
                            if latest_data_callback:
                                latest_data_callback(100)
                            print("Stitching process completed!")
                            return
                        break

    except Exception as e:
        print(f"Error in monitor_stitch_updates: {e}")
