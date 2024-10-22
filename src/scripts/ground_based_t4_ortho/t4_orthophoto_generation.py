import os
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.merge import merge
import numpy as np
import pandas as pd
import cv2
from affine import Affine as AffineTransform
from tqdm import tqdm 
from rasterio.io import MemoryFile
import gc
from osgeo import gdal
from utils import set_t4_ortho_progress

def update_consts(oddbed, geopath, source, temp):
    global ODD_BED
    global DISTANCE
    global GEO_PATH
    global SOURCE
    global TEMP_DIR
    ODD_BED = oddbed
    DISTANCE = 1.478133550530173e-05
    GEO_PATH = geopath
    SOURCE = source
    TEMP_DIR = temp

def crop_rotate_image(img, crop_amount):
    _, width = img.shape[:2]
    return cv2.flip(cv2.transpose(img[:, crop_amount : width - crop_amount])[::-1], 0)

def crop_rotate_image_odd(img, crop_amount):
    _, width = img.shape[:2]
    return cv2.flip(cv2.transpose(img[:, crop_amount : width - crop_amount]), -1)

def read_geo_file(file_path):
    df = pd.read_csv(file_path)
    df = df[['camA_file', 'lon_interp_adj', 'lat_interp_adj']]
    df.rename(columns={'camA_file': 'img_name', 
                       'lon_interp_adj': 'longitude', 
                       'lat_interp_adj': 'latitude'}, 
                       inplace=True)
    
    return df

def geoData(df):
    start_coords = (df.iloc[0].latitude, df.iloc[0].longitude)
    end_coords = (df.iloc[-1].latitude, df.iloc[-1].longitude)
    dx = end_coords[1] - start_coords[1]
    dy = end_coords[0] - start_coords[0]
    distance = DISTANCE
    angle = np.rad2deg(np.arctan2(dy, dx))

    return distance, angle

def calculate_end_coords(start_coords, distance, angle):
    lat1, lon1 = start_coords
    angle_rad = np.deg2rad(angle)
    
    delta_lon = distance * np.cos(angle_rad)
    delta_lat = distance * np.sin(angle_rad)
    
    end_lat = lat1 + delta_lat
    end_lon = lon1 + delta_lon
    
    return end_lat, end_lon

def process_image_odd(row, image_data, distance, angle, temp_dir):
    angle += 180
    height, width = image_data.shape[:2]

    start_coords = (row.latitude, row.longitude)
    end_coords = calculate_end_coords(start_coords, distance, angle)
    end_start_dist = np.sqrt((start_coords[1] - end_coords[1])**2 + (start_coords[0] - end_coords[0])**2)
    scaling_factor = end_start_dist / width
    translate = calculate_end_coords(start_coords, distance/3, angle)
    translate = calculate_end_coords(translate, distance*0.15, angle+90)

    # Affine transformation
    translation = AffineTransform.translation(translate[1], translate[0])
    rotation = AffineTransform.rotation(180 + angle)
    scaling = AffineTransform.scale(scaling_factor, scaling_factor)
    affine_transform = translation * rotation * scaling

    with MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
            count=3,
            dtype=image_data.dtype,
            width=width,
            height=height,
            crs='EPSG:4326',
            transform=affine_transform
        ) as src:
            for i in range(3):
                src.write(image_data[:, :, i], i + 1)

            # Recalculate transform and metadata for saving
            transform, new_width, new_height = calculate_default_transform(
                src.crs, 'EPSG:4326', width, height, *src.bounds)

            # Prepare final metadata
            final_metadata = src.meta.copy()
            final_metadata.update({
                'crs': 'EPSG:4326',
                'transform': transform,
                'width': new_width,
                'height': new_height,
                'nodata': 0
            })

            # Save to disk
            path = os.path.join(temp_dir, f"{row.img_name.split('.')[0]}_georef.tif")
            with rasterio.open(path, 'w', **final_metadata) as dst:
                for band in dst.indexes:
                    reproject(
                        source=rasterio.band(src, band),
                        destination=rasterio.band(dst, band),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs='EPSG:4326',
                        resampling=Resampling.nearest,
                        nodata=0
                    )

    return path

def process_image(row, image_data, distance, angle, temp_dir):
    angle += 180
    height, width = image_data.shape[:2]

    start_coords = (row.latitude, row.longitude)
    end_coords = calculate_end_coords(start_coords, distance, angle)
    end_start_dist = np.sqrt((start_coords[1] - end_coords[1])**2 + (start_coords[0] - end_coords[0])**2)
    scaling_factor = end_start_dist / width
    translate = calculate_end_coords(end_coords, -distance / 3, angle)
    translate = calculate_end_coords(translate, distance * 0.7, angle + 90)

    # Affine transformation
    translation = AffineTransform.translation(translate[1], translate[0])
    rotation = AffineTransform.rotation(180 + angle)
    scaling = AffineTransform.scale(scaling_factor, scaling_factor)
    affine_transform = translation * rotation * scaling

    with MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
            count=3,
            dtype=image_data.dtype,
            width=width,
            height=height,
            crs='EPSG:4326',
            transform=affine_transform
        ) as src:
            for i in range(3):
                src.write(image_data[:, :, i], i + 1)

            # Recalculate transform and metadata for saving
            transform, new_width, new_height = calculate_default_transform(
                src.crs, 'EPSG:4326', width, height, *src.bounds)

            # Prepare final metadata
            final_metadata = src.meta.copy()
            final_metadata.update({
                'crs': 'EPSG:4326',
                'transform': transform,
                'width': new_width,
                'height': new_height,
                'nodata': 0
            })

            # Save to disk
            path = os.path.join(temp_dir, f"{row.img_name.split('.')[0]}_georef.tif")
            with rasterio.open(path, 'w', **final_metadata) as dst:
                for band in dst.indexes:
                    reproject(
                        source=rasterio.band(src, band),
                        destination=rasterio.band(dst, band),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs='EPSG:4326',
                        resampling=Resampling.nearest,
                        nodata=0
                    )

    return path

def process_images(df, crop_amount, temp_dir, oddbed, geopath, source):
    update_consts(oddbed, geopath, source, temp_dir)
    distance, angle = geoData(df)
    os.makedirs(temp_dir, exist_ok=True)

    if not ODD_BED:
        for i, row in enumerate(tqdm(df.iterrows(), total=df.shape[0], desc='Processing images')):
            set_t4_ortho_progress((i+1.0)/df.shape[0] * 45)
            img_path = os.path.join(SOURCE, row.img_name)
            processed_img = crop_rotate_image(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), crop_amount)

            if processed_img is not None:
                process_image(row, processed_img, distance, angle, TEMP_DIR)

            del processed_img
            gc.collect()
    else:
        for i, row in tqdm(df.iterrows(), total=df.shape[0], desc='Processing images'):
            set_t4_ortho_progress((i+1.0)/df.shape[0] * 45)
            img_path = os.path.join(SOURCE, row.img_name)
            processed_img = crop_rotate_image_odd(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), crop_amount)

            if processed_img is not None:
                process_image_odd(row, processed_img, distance, angle, TEMP_DIR)

            del processed_img
            gc.collect()

    return True

def merge_georeferenced_images(output_path):
    os.makedirs("/".join(output_path.split("/")[:-1]), exist_ok=True)
    file_paths = [os.path.join(TEMP_DIR, file) for file in os.listdir(TEMP_DIR)]
    reprojected_processed = [rasterio.open(fp) for fp in tqdm(file_paths)]
    set_t4_ortho_progress(45 + 30)

    print("Merging...")
    mosaic, out_transform = merge(reprojected_processed)
    print("Done merging...")
    set_t4_ortho_progress(45 + 35)

    nodata_value = 0

    print("Updating metadata")
    metadata = reprojected_processed[0].meta.copy()
    metadata.update({
        'driver': 'GTiff',
        'count': mosaic.shape[0],
        'height': mosaic.shape[1],
        'width': mosaic.shape[2],
        'transform': out_transform,
        'nodata': nodata_value
    })

    print("Done updating metadata")
    set_t4_ortho_progress(45 + 37)

    with rasterio.open(output_path, 'w', **metadata) as dst:
        for i in range(mosaic.shape[0]):
            dst.write(mosaic[i], i + 1)

    for src in tqdm(reprojected_processed, desc='Closing files'):
        src.close()
    set_t4_ortho_progress(45 + 40)

    print(f"Combined GeoTIFF created at {output_path}")

def create_tiled_pyramid(input_path, output_path):
    try:
        ds = gdal.Open(input_path, gdal.GA_ReadOnly)
        if ds is None:
            raise FileNotFoundError(f"Unable to open input file: {input_path}")

        # Check the projection and geotransform information
        projection = ds.GetProjection()
        geotransform = ds.GetGeoTransform()
        print('Original Projection:', projection)
        print('GeoTransform:', geotransform)

        # Create a new raster in write mode with the same properties as the original raster
        driver = gdal.GetDriverByName('GTiff')
        if driver is None:
            raise RuntimeError("GTiff driver not available")

        dst_ds = driver.CreateCopy(output_path, ds, options=['TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW'])
        if dst_ds is None:
            raise RuntimeError(f"Failed to create output file: {output_path}")

        # Close the datasets to flush to disk
        ds = None
        dst_ds = None

        # Create the pyramid layers
        ds_pyramid = gdal.Open(output_path, gdal.GA_Update)
        if ds_pyramid is None:
            raise RuntimeError(f"Failed to reopen output file for updating: {output_path}")

        ds_pyramid.BuildOverviews('AVERAGE', [2, 4, 8, 16, 32, 64, 128, 256])
        ds_pyramid = None
        set_t4_ortho_progress(100)

    except Exception as e:
        print(f"An error occurred: {e}")


# def generate_orthophoto_T4(oddbed, geopath, source, temp, processed_path, date):
#     update_consts(oddbed, geopath, source, temp)
#     df = read_geo_file(GEO_PATH)
#     files = [f for f in os.listdir(SOURCE) if os.path.isfile(os.path.join(SOURCE, f))]
#     df = df[df.img_name.isin(files)]
#     process_images(df, 500, TEMP_DIR)
#     merge_georeferenced_images(processed_path + f'{date}-RGB.tif')
#     create_tiled_pyramid(processed_path + f'{date}-RGB.tif', processed_path + f'{date}-RGB-Pyramid.tif')

if __name__ == "__main__":
    print("debug")