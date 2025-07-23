import os
import sys
import re
import json
import math
import time
import shutil
import tempfile
import threading
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

def fit_angle_pca(x: np.ndarray, y: np.ndarray) -> float:
    pts = np.column_stack([x, y]) - np.column_stack([x, y]).mean(axis=0)
    _, _, v = np.linalg.svd(pts, full_matrices=False)
    vx, vy = v[0]
    return math.atan2(vy, vx)

def compute_axes_extents(xs: np.ndarray, ys: np.ndarray, theta: float, buffer_frac: float = 0.05):
    """Project (xs, ys) to heading axis (u) and orthogonal axis (v);
    return buffered extents and widths in meters."""
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    u_vec = np.array([cos_t, sin_t])        # along-track
    v_vec = np.array([-sin_t, cos_t])       # cross-track

    pts = np.column_stack([xs, ys])
    u_proj = pts @ u_vec
    v_proj = pts @ v_vec

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


def build_rotated_affine(u_min, v_max, width_m, height_m, theta, px, py):
    """Return rasterio Affine with rotation encoded."""
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    # top-left in (u,v)
    u_tl = u_min
    v_tl = v_max

    # back to XY (UTM)
    x_tl = u_tl * cos_t + v_tl * (-sin_t)
    y_tl = u_tl * sin_t + v_tl * ( cos_t)

    a =  px * cos_t
    b = -px * sin_t
    d =  py * sin_t
    e = -py * cos_t
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
    
def estimate_cross_track_auto(xs, ys, theta, h_pixels, px_along, strategy,
                              intr_gsd=None, fixed_min=None, ratio=None, k_sigma=5):
    """
    Returns (height_m_axis, v_min, v_max) based on chosen strategy.
    - strategy: 'intrinsics', 'std', 'ratio', 'fixed'
    """
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    v_vec = np.array([-sin_t, cos_t])
    v_vals = np.column_stack([xs, ys]) @ v_vec
    v_min = v_vals.min(); v_max = v_vals.max()
    height_gps = v_max - v_min

    if strategy == 'intrinsics' and intr_gsd is not None:
        height_need = intr_gsd * h_pixels
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
        v_center = 0.5 * (v_min + v_max)
        v_min = v_center - height_need / 2.0
        v_max = v_center + height_need / 2.0
        height_gps = height_need

    return height_gps, v_min, v_max

def reproject_to_wgs84(src_path: str, dst_path: str):
    print(f"Reprojecting {src_path} → {dst_path} (EPSG:4326)")
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, "EPSG:4326", src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update(dict(crs=CRS.from_epsg(4326),
                           transform=transform,
                           width=width,
                           height=height))
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

def georeference_plot(plot_id,
                      plot_data,
                      plot_stitched_path,
                      versioned_output_path,
                      has_stitch_direction,
                      camera_intrinsics=None):
    """
    Georeference a stitched plot image using GPS points at image centers.

    Modes:
      - GPS_ONLY:          only GPS extents, enforce a reasonable min cross-track (stats/ratio/fixed)
      - INTRINSICS_BOUND:  enforce min cross-track using intrinsics-derived GSD

    Keeps all your prints.
    """
    # ----------- CONFIG SWITCHES -----------
    MODE = "INTRINSICS_BOUND"      # "GPS_ONLY" or "INTRINSICS_BOUND"
    FORCE_SQUARE_PIXELS = True
    camera_height = 1.2            # meters (hard-coded)
    min_cross_track_aspect = 0.3   # floor = 30% of ideal (width * h/w) if GPS is too thin

    # Strategy options for helper
    CROSS_TRACK_STRATEGY = "intrinsics" if MODE == "INTRINSICS_BOUND" else "std"
    FIXED_MIN_CT = 0.6   # used if strategy == 'fixed'
    K_SIGMA = 5          # used if strategy == 'std'

    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        Image.MAX_IMAGE_PIXELS = None

        base_name = "AgRowStitch"
        print(f"\n========== Georeferencing Plot {plot_id} ==========")

        # ---- find mosaic ----
        mosaic_candidates = [
            f"{base_name}_plot-id-{plot_id}.png",
            f"{base_name}_plot-id-{plot_id}.tif",
            f"full_res_mosaic_temp_plot_{plot_id}.png",
            f"full_res_mosaic_temp_plot_{plot_id}.tif",
        ]
        src_file = None
        for fname in mosaic_candidates:
            p = os.path.join(versioned_output_path, fname)
            if os.path.exists(p):
                src_file = p
                break
        if src_file is None:
            print(f"[Plot {plot_id}] No mosaic file found in {versioned_output_path}")
            return False
        print(f"[Plot {plot_id}] Using mosaic file: {src_file}")

        # ---- GPS sanity ----
        if "lat" not in plot_data.columns or "lon" not in plot_data.columns:
            print(f"[Plot {plot_id}] lat/lon columns missing.")
            return False

        plot_df = (plot_data.sort_values("/top/rgb_file")
                   if "/top/rgb_file" in plot_data.columns
                   else plot_data.sort_index())

        lats = plot_df["lat"].dropna().to_numpy()
        lons = plot_df["lon"].dropna().to_numpy()
        if lats.size == 0 or lons.size == 0:
            print(f"[Plot {plot_id}] No valid lat/lon values.")
            return False

        first_lat, first_lon = lats[0], lons[0]
        last_lat,  last_lon  = lats[-1], lons[-1]
        print(f"[Plot {plot_id}] First GPS  ({first_lat:.6f}, {first_lon:.6f})")
        print(f"[Plot {plot_id}] Last  GPS  ({last_lat:.6f}, {last_lon:.6f})")

        lat_diff = last_lat - first_lat
        lon_diff = last_lon - first_lon
        if abs(lat_diff) > abs(lon_diff):
            rover_direction = "NORTH" if lat_diff > 0 else "SOUTH"
        else:
            rover_direction = "EAST" if lon_diff > 0 else "WEST"
        print(f"[Plot {plot_id}] Rover direction guess: {rover_direction}")

        if has_stitch_direction and "stitch_direction" in plot_df.columns:
            stitch_direction = str(plot_df["stitch_direction"].iloc[0]).upper()
        else:
            stitch_direction = "UNKNOWN"
        print(f"[Plot {plot_id}] Stitching direction (column): {stitch_direction}")

        # ---- read mosaic ----
        with Image.open(src_file) as im:
            if im.mode != "RGB":
                im = im.convert("RGB")
            img_array = np.array(im)
        h, w = img_array.shape[:2]
        print(f"[Plot {plot_id}] Image size: {w} x {h}")

        # ---- choose UTM ----
        center_lat = (lats.min() + lats.max()) / 2.0
        center_lon = (lons.min() + lons.max()) / 2.0
        utm_epsg = pick_utm_epsg(center_lon, center_lat)
        print(f"[Plot {plot_id}] Using UTM CRS: EPSG:{utm_epsg}")

        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
        xs, ys = transformer.transform(lons, lats)

        # ---- PCA heading ----
        theta = fit_angle_pca(xs, ys)
        theta_deg = math.degrees(theta) % 360
        print(f"[Plot {plot_id}] PCA trajectory angle: {theta_deg:.2f}° (0° = +X/East)")

        # enforce left→right order (first should be left)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        u_first = cos_t * xs[0] + sin_t * ys[0]
        u_last  = cos_t * xs[-1] + sin_t * ys[-1]
        if u_last < u_first:
            theta += math.pi
            theta_deg = (theta_deg + 180) % 360
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            print(f"[Plot {plot_id}] Flipped θ by 180° to match mosaic L→R order")

        # ---- axis extents from GPS ----
        u_min, u_max, v_min, v_max, width_m_axis, height_m_axis = compute_axes_extents(
            xs, ys, theta, buffer_frac=0.05
        )
        print(f"[Plot {plot_id}] Raw axis extents: width={width_m_axis:.3f}m, height={height_m_axis:.3f}m")

        # ---- cross-track decision using helper ----
        px_along = width_m_axis / w  # meters per pixel along-track

        intr_gsd = None
        if CROSS_TRACK_STRATEGY == "intrinsics" and camera_intrinsics:
            intr_gsd = calc_gsd_from_intrinsics(camera_intrinsics, camera_height, w, h)
            if intr_gsd is not None:
                print(f"[Plot {plot_id}] Intrinsics GSD: {intr_gsd:.6f} m/px")

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
            k_sigma=K_SIGMA
        )

        if height_m_axis < min_height_allowed:
            print(f"[Plot {plot_id}] Cross-track too small ({height_m_axis:.3f}m). Expanding to {min_height_allowed:.3f}m.")
            v_center = 0.5 * (v_min + v_max)
            v_min = v_center - min_height_allowed / 2.0
            v_max = v_center + min_height_allowed / 2.0
            height_m_axis = min_height_allowed

        # ---- pixel sizes ----
        px = width_m_axis / w
        py = height_m_axis / h
        if FORCE_SQUARE_PIXELS:
            px = py = max(px, py)
            width_m_axis  = px * w
            height_m_axis = py * h
            v_center = 0.5 * (v_min + v_max)
            v_min = v_center - height_m_axis / 2.0
            v_max = v_center + height_m_axis / 2.0
            print(f"[Plot {plot_id}] Forced square pixels. px=py={px:.6f} m/px")

        gsd_avg = (px + py) / 2.0
        print(f"[Plot {plot_id}] Final ground coverage: {width_m_axis:.2f}m x {height_m_axis:.2f}m")
        print(f"[Plot {plot_id}] Pixel size / GSD: {gsd_avg:.6f} m/px (x: {px:.6f}, y: {py:.6f})")

        # ---- affine transform ----
        transform = build_rotated_affine(u_min, v_max, width_m_axis, height_m_axis, theta, px, py)

        print("[Plot {plot_id}] Affine transform:")
        print(f"    a={transform.a:.8f}, b={transform.b:.8f}, c={transform.c:.3f}")
        print(f"    d={transform.d:.8f}, e={transform.e:.8f}, f={transform.f:.3f}")
        print(f"[Plot {plot_id}] Rotation encoded: {theta_deg:.2f}°; rover guess: {rover_direction}")

        # ---- write UTM GeoTIFF ----
        utm_out = os.path.join(versioned_output_path, f"georeferenced_plot_{plot_id}_utm.tif")
        print(f"[Plot {plot_id}] Writing UTM GeoTIFF: {utm_out}")
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
        print(f"[Plot {plot_id}] UTM GeoTIFF written.")

        # ---- reproject to WGS84 ----
        wgs_out = os.path.join(versioned_output_path, f"georeferenced_plot_{plot_id}.tif")
        reproject_to_wgs84(utm_out, wgs_out)

        print(f"[Plot {plot_id}] DONE. Outputs:")
        print(f"    UTM GeoTIFF : {utm_out}")
        print(f"    WGS84 GeoTIFF: {wgs_out}")
        print("====================================================\n")
        return True

    except Exception as e:
        print(f"[Plot {plot_id}] ERROR during georeferencing: {e}")
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
        required_columns = ['plot_id', '/top/rgb_file']
        missing_columns = [col for col in required_columns if col not in msgs_df.columns]
        if missing_columns:
            print(f"Error: Missing columns in msgs_synced.csv: {missing_columns}")
            return
        
        # Get unique plot IDs
        unique_plots = msgs_df['plot_id'].unique()
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
        
        for plot_id in unique_plots:
            if stitch_stop_event and stitch_stop_event.is_set():
                print("Stitching process stopped by user")
                break
            
            # Skip plot index 0
            if plot_id == 0:
                print(f"Skipping plot {plot_id} (plot index 0)")
                continue
                
            try:
                print(f"Processing plot {plot_id}")
                
                # Filter data for current plot
                plot_data = msgs_df[msgs_df['plot_id'] == plot_id]

                # Get image filenames for this plot
                image_files = plot_data['/top/rgb_file'].dropna().tolist()
                if not image_files:
                    print(f"No images found for plot {plot_id}, skipping")
                    failed_plots.append(plot_id)
                    continue
                
                # Create temporary directory for this plot's images
                plot_temp_dir = os.path.join(os.path.dirname(image_path), f"temp_plot_{plot_id}")
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
                    print(f"No valid images copied for plot {plot_id}, skipping")
                    shutil.rmtree(plot_temp_dir)
                    failed_plots.append(plot_id)
                    continue
                
                print(f"Copied {len(copied_files)} images for plot {plot_id}")
                
                # Load original config
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)

                # Update fields for this plot
                config['image_directory'] = plot_temp_dir
                config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
                
                # Set stitching direction if available
                if has_stitch_direction:
                    stitching_direction = plot_data['stitch_direction'].iloc[0]
                    print(f"Stitching direction for plot {plot_id}: {stitching_direction}")
                    if pd.notna(stitching_direction):
                        config['stitching_direction'] = stitching_direction
                        print(f"Set stitching_direction to {stitching_direction} for plot {plot_id}")
                else:
                    print(f"No stitching_direction found for plot {plot_id}, using default direction")

                # Apply any user-specified custom options
                if isinstance(custom_options, str):
                    custom_options = yaml.safe_load(custom_options)
                if custom_options and isinstance(custom_options, dict):
                    config.update(custom_options)

                # Save to temp config file
                with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix=f"_plot_{plot_id}.yaml") as tmpfile:
                    yaml.safe_dump(config, tmpfile)
                    temp_config_path = tmpfile.name

                cpu_count = os.cpu_count()
                
                # Create plot-specific output paths - use the shared final_mosaics directory
                plot_stitched_path = os.path.join(os.path.dirname(image_path), "final_mosaics")
                
                # Run stitching for this plot
                print(f"Starting stitching for plot {plot_id}")
                
                # Run the stitching process for this plot
                run_stitch_process_for_plot(temp_config_path, cpu_count, plot_stitched_path, versioned_output_path, plot_id, stitch_stop_event)
                
                # Georeference the individual plot at full resolution (maximum quality)
                print(f"Georeferencing individual plot {plot_id} at full resolution...")
                georeference_success = georeference_plot(plot_id, plot_data, plot_stitched_path, versioned_output_path, has_stitch_direction, camera_intrinsics)
                if georeference_success:
                    print(f"Successfully georeferenced plot {plot_id}")
                else:
                    print(f"Warning: Failed to georeference plot {plot_id}")
                
                # Clean up temporary files
                os.unlink(temp_config_path)
                shutil.rmtree(plot_temp_dir)
                
                processed_plots.append(plot_id)
                print(f"Successfully processed plot {plot_id}")
                
                # DEBUG: Break after plot 3 and create combined mosaic # TODO: DELETE LATE
                if plot_id == 3:
                    print("DEBUG: Breaking after plot 3 to create combined mosaic")
                    break
                
            except Exception as e:
                print(f"Error processing plot {plot_id}: {str(e)}")
                failed_plots.append(plot_id)
                
                # Clean up on error
                if 'plot_temp_dir' in locals() and os.path.exists(plot_temp_dir):
                    shutil.rmtree(plot_temp_dir)
                if 'temp_config_path' in locals() and os.path.exists(temp_config_path):
                    os.unlink(temp_config_path)

        result_message = f"Stitching completed. Processed plots: {processed_plots}"
        if failed_plots:
            result_message += f", Failed plots: {failed_plots}"
        
        print(result_message)
        
        # Skip combined mosaic creation - individual georeferenced TIFFs provide maximum quality
        if processed_plots:
            print(f"Successfully processed and georeferenced {len(processed_plots)} individual plots at full resolution")
            print("Individual georeferenced TIFF files created for maximum quality - no combined mosaic needed")
            if final_progress_callback:
                final_progress_callback(100)  # Completed everything
        else:
            print("No plots were successfully processed")
        
    except Exception as e:
        print(f"Error in run_stitch_all_plots: {str(e)}")


def run_stitch_process_for_plot(config_path, cpu_count, stitched_path, versioned_output_path, plot_id, stitch_stop_event=None):
    """Run the stitching process for a single plot"""
    print(f"Stitching started for plot {plot_id} on thread {threading.current_thread().name}")

    try:
        # Run the AgRowStitch process
        print(f"Calling run_agrowstitch with config: {config_path} and cpu_count: {cpu_count}")
        agrowstitch_result = run_agrowstitch(config_path, cpu_count)
        print(f"AgRowStitch result type: {type(agrowstitch_result)}")
        print(f"AgRowStitch result: {agrowstitch_result}")
        
        # Check if the result is iterable
        if agrowstitch_result is None:
            print(f"AgRowStitch returned None for plot {plot_id}")
            # Don't return immediately, continue to check for output files
        else:
            # Try to iterate through the result
            try:
                if hasattr(agrowstitch_result, '__iter__') and not isinstance(agrowstitch_result, (str, bytes)):
                    for step in agrowstitch_result:
                        if stitch_stop_event and stitch_stop_event.is_set():
                            print(f"Stitching canceled by user for plot {plot_id}.")
                            return
                        print(f"Plot {plot_id} - Step: {step}")
                else:
                    print(f"AgRowStitch result is not iterable for plot {plot_id}, but process may have completed")
            except TypeError as e:
                print(f"AgRowStitch result is not iterable for plot {plot_id}: {e}")
                # If it's not iterable, assume the process completed successfully
                print(f"Assuming stitching completed successfully for plot {plot_id}")
            except Exception as e:
                print(f"Error during AgRowStitch iteration for plot {plot_id}: {e}")
                # Don't return, continue to check for output files
        
        print(f"AgRowStitch processing completed for plot {plot_id}, checking for output files...")

        # Now stitching is done — look for the actual generated files
        # Check if the stitched_path exists, if not wait a bit for it to be created
        wait_count = 0
        max_wait_attempts = 30  # Increased wait time
        while not os.path.exists(stitched_path) and wait_count < max_wait_attempts:
            print(f"Waiting for stitched output directory: {stitched_path} (attempt {wait_count + 1}/{max_wait_attempts})")
            time.sleep(5)  # Increased wait time per attempt
            wait_count += 1
        
        if not os.path.exists(stitched_path):
            print(f"Stitched output directory not found for plot {plot_id}: {stitched_path}")
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
                    plot_files = [f for f in all_files if str(plot_id) in f or 'plot' in f.lower()]
                    if plot_files:
                        stitched_path = alt_dir
                        print(f"Using alternative directory {alt_dir} with files: {plot_files}")
                        break
            else:
                print(f"No suitable output directory found for plot {plot_id}")
                return
        
        # List all files in the stitched directory to see what was actually generated
        all_files = os.listdir(stitched_path)
        print(f"Files found in {stitched_path}: {all_files}")
        
        # Look for files that match this plot_id pattern - try multiple patterns
        # Only use plot-specific patterns first, avoid generic ones that could match multiple plots
        plot_patterns = [
            f"temp_plot_{plot_id}",
            f"full_res_mosaic_temp_plot_{plot_id}",
            f"plot_{plot_id}",
            f"plot-{plot_id}",
            str(int(plot_id)) if float(plot_id).is_integer() else str(plot_id),
        ]
        
        plot_files = []
        for pattern in plot_patterns:
            matching_files = [f for f in all_files if pattern in f]
            plot_files.extend(matching_files)
        
        # Remove duplicates while preserving order
        plot_files = list(dict.fromkeys(plot_files))
        print(f"Files matching plot {plot_id} patterns: {plot_files}")
        
        # If no specific plot files found, look for any image files that might be the result
        if not plot_files:
            image_extensions = ['.png', '.tif', '.tiff', '.jpg', '.jpeg']
            image_files = [f for f in all_files if any(f.lower().endswith(ext) for ext in image_extensions)]
            print(f"No plot-specific files found. Available image files: {image_files}")
            
            # If there's only one image file, assume it's our result
            if len(image_files) == 1:
                plot_files = image_files
                print(f"Using single image file as result: {plot_files}")
            elif len(image_files) > 1:
                # Look for files with "mosaic" or "stitch" in the name AND the plot_id
                plot_specific_mosaic = [f for f in image_files if 
                                      ('mosaic' in f.lower() or 'stitch' in f.lower()) and 
                                      str(plot_id) in f]
                if plot_specific_mosaic:
                    plot_files = plot_specific_mosaic
                    print(f"Using plot-specific mosaic files: {plot_files}")
                else:
                    # Only use generic approach if we're sure it's plot-specific
                    # Look for files that contain the plot_id as a substring
                    plot_id_files = [f for f in image_files if str(plot_id) in f]
                    if plot_id_files:
                        # Take the largest plot-specific file
                        largest_file = max(plot_id_files, key=lambda f: os.path.getsize(os.path.join(stitched_path, f)))
                        plot_files = [largest_file]
                        print(f"Using largest plot-specific image file: {plot_files}")
                    else:
                        print(f"Warning: No plot-specific files found for plot {plot_id}. Skipping to avoid incorrect georeferencing.")
                        return
        
        if not plot_files:
            print(f"No files found for plot {plot_id} in {stitched_path}")
            return
        
        # Copy files directly to the versioned output directory
        base_name = "AgRowStitch"
        main_mosaic_file = None
        files_copied = 0
        
        for file_name in plot_files:
            try:
                src_file_path = os.path.join(stitched_path, file_name)
                
                # Verify source file exists and has content
                if not os.path.exists(src_file_path):
                    print(f"Source file does not exist: {src_file_path}")
                    continue
                    
                file_size = os.path.getsize(src_file_path)
                if file_size == 0:
                    print(f"Source file is empty: {src_file_path}")
                    continue
                    
                print(f"Copying {file_name} (size: {file_size} bytes)")
                
                # Create destination file path
                dst_file_path = os.path.join(versioned_output_path, file_name)
                
                # Copy the file
                shutil.copy2(src_file_path, dst_file_path)
                print(f"Successfully copied {file_name} to {versioned_output_path}")
                files_copied += 1
                
                # Identify the main mosaic file for renaming
                # Look for patterns that indicate the main mosaic AND contain the plot_id
                main_patterns = [
                    f"temp_plot_{plot_id}",
                    f"full_res_mosaic_temp_plot_{plot_id}",
                ]
                
                # Also check for generic patterns but only if plot_id is in filename
                generic_patterns = ["mosaic", "stitch"]
                
                # First try plot-specific patterns
                is_main_file = any(pattern in file_name.lower() for pattern in main_patterns)
                
                # If not found, try generic patterns but only if plot_id is in filename
                if not is_main_file and str(plot_id) in file_name:
                    is_main_file = any(pattern in file_name.lower() for pattern in generic_patterns)
                
                if is_main_file:
                    main_mosaic_file = file_name
                    
            except Exception as e:
                print(f"Error copying file {file_name}: {str(e)}")
                continue
        
        print(f"Successfully copied {files_copied} files for plot {plot_id}")
        
        if files_copied == 0:
            print(f"No files were successfully copied for plot {plot_id}")
            return
        
        # Rename the main mosaic file to include a cleaner name
        if main_mosaic_file:
            try:
                src_mosaic = os.path.join(versioned_output_path, main_mosaic_file)
                file_extension = os.path.splitext(main_mosaic_file)[1]
                new_mosaic_name = f"{base_name}_plot-id-{plot_id}{file_extension}"
                dst_mosaic = os.path.join(versioned_output_path, new_mosaic_name)
                
                # Only rename if the destination doesn't already exist
                if not os.path.exists(dst_mosaic):
                    shutil.move(src_mosaic, dst_mosaic)
                    print(f"Renamed main mosaic to {new_mosaic_name}")
                else:
                    print(f"Destination file already exists, skipping rename: {new_mosaic_name}")
            except Exception as e:
                print(f"Error renaming main mosaic file: {str(e)}")
        else:
            print(f"Warning: No main mosaic file found to rename for plot {plot_id}")
        
        print(f"Completed processing plot {plot_id} - copied {files_copied} files to {versioned_output_path}")
            
    except Exception as e:
        print(f"Error in stitching process for plot {plot_id}: {str(e)}")
        raise


def monitor_stitch_updates_multi_plot(final_mosaics_dir, processed_plots, latest_data_callback=None):
    """Monitor multiple plot log files and aggregate progress"""
    try:
        # Wait for the final_mosaics directory to be created
        while not os.path.exists(final_mosaics_dir):
            print("Waiting for final_mosaics directory to be created...")
            time.sleep(5)

        print(f"Monitoring log files in: {final_mosaics_dir}")
        
        # Track completed plots
        completed_plots = set()
        total_plots = len(processed_plots)
        
        while True:
            # Check each plot's log file to see if it's completed
            for plot_id in processed_plots:
                if plot_id not in completed_plots:
                    log_file = os.path.join(final_mosaics_dir, f"temp_plot_{plot_id}.log")
                    
                    if os.path.exists(log_file):
                        # Check if this plot is done by looking for completion indicators
                        if is_plot_completed(log_file):
                            completed_plots.add(plot_id)
                            print(f"Plot {plot_id} completed!")
            
            # Calculate progress based on completed plots (0-100% since no combined mosaic)
            progress_percentage = int((len(completed_plots) / total_plots) * 100)
            
            # Update progress using callback if provided
            if latest_data_callback:
                latest_data_callback(progress_percentage)
            
            # Check if all plots are complete
            if len(completed_plots) == total_plots:
                print("All individual plots completed and georeferenced!")
                if latest_data_callback:
                    latest_data_callback(100)
                break
                
            time.sleep(3)  # Check every 3 seconds

    except Exception as e:
        print(f"Error in log monitoring: {e}")


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
