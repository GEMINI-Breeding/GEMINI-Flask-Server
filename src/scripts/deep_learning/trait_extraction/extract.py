import argparse
import torch
import cv2
import os
import skimage
import ast
import re
import warnings
import math
import gc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd
import albumentations as A

from cv2 import cuda
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
from osgeo import gdal, osr
from shapely.geometry import Polygon
from locate import create_stereo, parse_dir, get_camera_params, \
                split_into_chunks, StereoWrapper, process_disparity
from tqdm.contrib.concurrent import process_map
from typing import List

warnings.filterwarnings('ignore', category=UserWarning)

# global
PIXEL_AREA = 0.00667 # cm2 ## NOTE: automate this
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 6

class ImageData:

    def __init__(self, outs, tif_arrs, geos, widths, heights, ts, summ_df, emerging, plotmap):
        # parameters
        self.ts = ts
        self.ts_dfs = summ_df
        self.plotmap = gpd.read_file(plotmap)
        self.plotmap = self.plotmap.to_crs(self.plotmap.estimate_utm_crs())
        # self.p = p

        # load out files
        self.tif_arr = tif_arrs
        self.geo = geos
        self.width = widths
        self.height = heights
        self.dpth_map = outs

        # crop depth
        self.dpth_map = self.crop() ##NOTE: CROP

        # convert tif to store geographical coordinates
        self.tif_coords = self.convert_to_geo()
        
        # extract ts bed and tier
        if emerging:
            bed_value = self.ts_dfs[self.ts_dfs['Timestamp'] == self.ts]['Bed'].tolist()[0]
            self.bed = int(bed_value) if not np.isnan(bed_value) else 1  # replace 0 with your default value
            tier_value = self.ts_dfs[self.ts_dfs['Timestamp'] == self.ts]['Tier'].tolist()[0]
            self.tier = int(tier_value) if not np.isnan(tier_value) else 1  # replace 0 with your default value
        else:
            self.bed, self.tier = self.match_to_plots()

    def crop(self):
        # crop
        dpth_map = self.dpth_map[:,600:-600]
        return dpth_map
    
    def convert_to_geo(self):
        # convert
        geo = self.geo
        width = self.width
        height = self.height
        
        # create arrays representing X and Y coordinates of each pixel
        x_indices, y_indices = np.meshgrid(np.arange(width), np.arange(height))
        
        # apply the geotransform
        eastings = geo[0] + x_indices * geo[1] + y_indices * geo[2]
        northings = geo[3] + x_indices * geo[4] + y_indices * geo[5]
        
        # stack the easting and northing coordinates
        tif_coords = np.dstack((eastings, northings))
        
        return tif_coords
    
    def match_to_plots(self):
        # extract image polygon
        x_origin, pixel_width, _, y_origin, _, pixel_height = self.geo
        top_left = (x_origin, y_origin); # top_left = self.p(top_left[0], top_left[1], inverse=True)
        top_right = (x_origin + self.width * pixel_width, y_origin); # top_right = self.p(top_right[0], top_right[1], inverse=True)
        bottom_left = (x_origin, y_origin + self.height * pixel_height); # bottom_left = self.p(bottom_left[0], bottom_left[1], inverse=True)
        bottom_right = (x_origin + self.width * pixel_width, y_origin + self.height * pixel_height); # bottom_right = self.p(bottom_right[0], bottom_right[1], inverse=True)
        polygon = Polygon([top_left, top_right, bottom_right, bottom_left, top_left])
        
        # choose the plot the polygon overlaps the most with
        self.plotmap['intersection_area'] = self.plotmap['geometry'].apply(lambda x: x.intersection(polygon).area if x.intersects(polygon) else 0)
        max_overlap_index = self.plotmap['intersection_area'].idxmax()
        
        # return tier and bed
        if self.plotmap.loc[max_overlap_index, 'intersection_area'] > 0:
            # Retrieve the Bed and Tier values for the row with maximum overlap
            bed = self.plotmap.loc[max_overlap_index, 'column']
            tier = self.plotmap.loc[max_overlap_index, 'row']
            return bed, tier
        else:
            return None, None

def filter_bboxes(
    preds: dict, 
    region: np.ndarray
) -> dict:
    
    # preds_filtered = {}
        
    # flatten region
    region = np.array(region).flatten()
    
    # iterate through each predictions
    # keys = list(preds.keys())
    # for key in keys:
        
    # boxes = preds[key]
    boxes = preds
    box_list = []
    for box in boxes:
        x1, y1, x2, y2 = box # get coordinates
        
        # check if box is within region
        if x1 <= region[2] and x2 >= region[0] and y1 <= region[3] and y2 >= region[1]:
            
            # update box list
            box_list.append(box)

        # update dictionary
        # preds_filtered[key] = box_list
        
    return box_list

def postprocessing(
    args: List
) -> pd.DataFrame:

    image, preds, bboxes, indices, summ_df, trait, skip_stereo = args

    # return image to original size (and boxes) ##NOTE: CROP
    if not skip_stereo:
        transform = A.Compose([A.CenterCrop(height=2048, width=1392, p=1.0)],
                        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        t = transform(image=image, bboxes=preds, class_labels=[1]*len(preds))

    # loop through coordinates
    for i in range(len(bboxes)):
        bboxes_filt = filter_bboxes(t['bboxes'], bboxes[i])
        
        # count number of traits
        count = len(bboxes_filt)

        # update summary dataframe
        summ_df.loc[indices[i], f'{trait.capitalize()} Count'] = int(count)

    return summ_df

def predict(
    images_batch: List[np.ndarray], 
    summs_batch: List[pd.DataFrame], 
    indices_batch: List[int], 
    bboxes_batch: List[np.ndarray], 
    model: YOLO,
    trait: str,
    skip_stereo: bool
) -> pd.DataFrame:

    predictions = dict()
    # pod_model, leaf_model, flower_model = models_batch

    # run prediction for pods
    if 'pod' in trait.lower():
        results = model.predict(images_batch, iou=0.3, device=0, verbose=False)
        predictions[trait] = [result.boxes.xyxy.cpu().numpy() for result in results]
    elif 'leaf' in trait.lower():
        # run prediction for leaf
        results = model.predict(images_batch, iou=0.2, conf=0.1, device=0, verbose=False)
        predictions[trait] = [result.boxes.xyxy.cpu().numpy() for result in results]
    elif 'flower' in trait.lower():
        # run prediction for flower
        results = model.predict(images_batch, iou=0.5, device=0, verbose=False)
        predictions[trait] = [result.boxes.xyxy.cpu().numpy() for result in results]

    # run postprocessing on map
    post_args = list(zip(
        images_batch,
        predictions[trait],
        bboxes_batch,
        indices_batch,
        summs_batch,
        [trait]*len(images_batch),
        [skip_stereo]*len(images_batch)
    ))
    df = list(process_map(postprocessing, post_args, max_workers=num_workers, chunksize=1, disable=True))

    return df

def calc_height_area(dpth_map, coord, floor=-170):
    crop = dpth_map[coord[0][1]:coord[1][1], coord[0][0]:coord[1][0]]
    crop_hghts = crop[crop < -60]
    crop_hghts = crop_hghts[crop_hghts < 0] - floor
    crop_hghts = crop_hghts[crop_hghts > 0]

    height = np.mean(crop_hghts)
    volume = np.size(crop_hghts)*height*PIXEL_AREA
    
    return height, volume

def match_depth_color(tif_arr, dpth_map, summ_df, indices, coords, emerging):

    # mask out plant
    hsv_arr = skimage.color.rgb2hsv(tif_arr)
    hsv_arr = (hsv_arr*255.).astype(int)
    mask_h = hsv_arr[:,:,0] > 40
    mask_s = hsv_arr[:,:,1] > 15
    mask_v = hsv_arr[:,:,2] > 15
    mask = mask_h*mask_s*mask_v # True is leaf
    mask = skimage.morphology.opening(mask, skimage.morphology.diamond(3)) > 0

    # apply mask to depth map and depth array
    dpth_map = dpth_map[:,:,2]
    dpth_map *= mask

    for i in range(len(coords)):
            
        # calculate height and area
        coord = coords[i]
        height, volume = calc_height_area(dpth_map, coord)

        # update summary dataframe
        summ_df.loc[indices[i], 'Height (cm)'] = height
        summ_df.loc[indices[i], 'Area (cm2)'] = volume/height

    return summ_df

def partition(
    img_coords: List, 
    img: ImageData
) -> List:

    # get center coordinates
    center_coords = []
    for coord in img_coords:
        x1, y1 = coord[0]
        x2, y2 = coord[1]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        center_coords.append([int(center_x), int(center_y)])

    # partition coordinates
    boundary = int(img.tif_arr.shape[1]/2)
    centers_1 = np.array([coord for coord in center_coords if coord[0] < boundary])
    centers_2 = np.array([coord for coord in center_coords if coord[0] > boundary])

    # add line seperations
    lines_1 = []
    if len(centers_1) > 1:
        y_1 = np.sort(centers_1[:,1], axis=0)
        for i in range(len(y_1)):
            y = ((y_1[i+1] - y_1[i]) / 2) + y_1[i]
            lines_1.append(int(y))
            if i+1 == len(y_1)-1:
                break

    lines_2 = []
    if len(centers_2) > 1:
        y_2 = np.sort(centers_2[:,1], axis=0)
        for i in range(len(y_2)):
            y = ((y_2[i+1] - y_2[i]) / 2) + y_2[i]
            lines_2.append(int(y))
            if i+1 == len(y_2)-1:
                break

    # add borders
    if len(centers_1) >= 1:
        lines_1.append(0)
        lines_1.append(img.tif_arr.shape[0])
    if len(centers_2) >= 1:
        lines_2.append(0)
        lines_2.append(img.tif_arr.shape[0])

    # create bounding boxes per section
    lines_1 = np.array(lines_1)
    lines_1 = np.sort(lines_1)
    bboxes_1 = []
    for i in range(len(lines_1)):
        box = [[0, lines_1[i]], [boundary, lines_1[i+1]]]
        bboxes_1.append(box)

        if i+1 == len(lines_1)-1:
            break
    lines_2 = np.array(lines_2)
    lines_2 = np.sort(lines_2)
    bboxes_2 = []
    for i in range(len(lines_2)):
        box = [[boundary, lines_2[i]], [img.tif_arr.shape[1], lines_2[i+1]]]
        bboxes_2.append(box)

        if i+1 == len(lines_2)-1:
            break

    return bboxes_1 + bboxes_2

def extract(
    dim_args: List
) -> List:

    # initialize image properties
    outs, tif_arrs, geos, widths, heights, ts, summ_df, emerging, plotmap, trait, skip_stereo = dim_args
    img = ImageData(outs, tif_arrs, geos, widths, heights, ts, summ_df, emerging, plotmap)
    image = cv2.cvtColor(img.tif_arr, cv2.COLOR_BGR2RGB)

    # load summary
    summ_df['Height (cm)'] = np.nan
    summ_df['Area (cm2)'] = np.nan
    summ_df[f'{trait.capitalize()} Count'] = np.nan

    # filter summary
    summ_filt = filter_df(summ_df, img)

    # get image coordinates from these boxes
    img_coords = []
    indices = []
    for index, row in summ_filt.iterrows():
        coords = [[row['UTM'][0], row['UTM'][1]],[row['UTM'][2], row['UTM'][3]]]

        for i in range(len(coords)):

            dist = np.linalg.norm(img.tif_coords - coords[i], axis=2)
            closest_idx = np.unravel_index(np.argmin(dist), dist.shape)
            coords[i] = list(closest_idx)
        indices.append(row['index'])
        img_coords.append([[coords[0][1],coords[0][0]], [coords[1][1],coords[1][0]]])

    # calculate dimensions
    if emerging:
        # update summary dataframe
        bboxes = img_coords
        summ_df = match_depth_color(img.tif_arr, img.dpth_map, \
            summ_df, indices, img_coords, emerging)
    else:
        bboxes = partition(img_coords, img)
        summ_df = match_depth_color(img.tif_arr, img.dpth_map, \
            summ_df, indices, bboxes, emerging)

    # preprare transformation for prediction ##NOTE: CROP
    if not skip_stereo:
        transform = A.PadIfNeeded(min_height=2048, min_width=2592, 
                pad_height_divisor=None, pad_width_divisor=None, border_mode=0, value=(0, 0, 0), mask_value=None)
        image = transform(image=image)['image']

    return [image, summ_df, indices, bboxes]

def filter_df(
    summ_df: pd.DataFrame, 
    img: ImageData
) -> pd.DataFrame:

    # filter summary
    summ_filt = summ_df[(summ_df['Bed'] == img.bed) & (summ_df['Tier'] == img.tier)]
    summ_filt = summ_filt.reset_index()
    summ_filt['UTM'] = summ_filt['UTM'].apply(lambda x: np.array(ast.literal_eval(x)))

    return summ_filt

def tiffout(
    args: List
) -> List:
    """
    Convert image into tif containing geo coordinates per pixel
    """

    out3d, images, ts, camera, skip_stereo = args

    try:
        filename = f'{camera}-{ts}.jpg'
        path = next((path for path in images if path.endswith(filename)), None)
        rgb_im = plt.imread(path)
    except Exception as e:
        print(f'Failed to read image path: {path}')
        print(e)
        return 0

    #  Initialize the Image Size ##NOTE: CROP
    if not skip_stereo:
        try:
            out3d = out3d[:,600:-600,:]
            rgb_im = rgb_im[:,600:-600,:]
        except Exception as e:
            print(type(out3d))
    image_size = (out3d.shape[:2])

    # set geotransform
    nx = image_size[1]
    ny = image_size[0]
    xmin, ymin, xmax, ymax = np.array([np.min(out3d[:,:,0]), np.min(out3d[:,:,1]), 
                                        np.max(out3d[:,:,0]), np.max(out3d[:,:,1])])/100
    xres = (xmax - xmin) / float(nx)
    yres = (ymax - ymin) / float(ny)
    geotransform = (xmin, xres, 0, ymax, 0, -yres) # start top left, move right (+x) and down (-y)

    # Create a memory dataset instead of a TIFF file
    mem_drv = gdal.GetDriverByName('MEM')
    dst_ds = mem_drv.Create('', nx, ny, 3, gdal.GDT_Byte)
    
    dst_ds.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()            
    srs.ImportFromEPSG(32610)                
    dst_ds.SetProjection(srs.ExportToWkt())
    
    # Write data to memory dataset
    dst_ds.GetRasterBand(1).WriteArray(rgb_im[:, :, 0])
    dst_ds.GetRasterBand(2).WriteArray(rgb_im[:, :, 1])
    dst_ds.GetRasterBand(3).WriteArray(rgb_im[:, :, 2])
    
    # Convert memory dataset to NumPy array
    geo_t = dst_ds.GetGeoTransform()
    width = dst_ds.RasterXSize
    height = dst_ds.RasterYSize
    
    dst_ds = None  # This will close the dataset

    tiffouts = [rgb_im, geo_t, width, height]
    
    return tiffouts
    
def split_into_batches(data_list, batch_size):
    return [data_list[i:i + batch_size] for i in range(0, len(data_list), batch_size)]

def unpack_batch(batch):
    return list(zip(*batch))

def update_progress(
    save: Path,
    task: str
) -> int:
    """Update progress based on task accomplished."""
    scores = {'configs': 1,'models': 2,'images': 4, \
                'filter': 95,'export': 100}
    progress = scores[task]
    
    # write in text file
    with open(save.parent/'extract_progress.txt', 'w') as file:
        file.write(str(progress))

def main(
    emerging: bool,
    summary: Path,
    images_path: Path,
    camera: str,
    camera_stereo: List[str],
    plotmap: Path,
    batch_size: int,
    model_path: Path,
    metadata: Path,
    save: Path,
    temp: Path,
    trait: str,
    skip_stereo: bool,
    geojson_filename: str
) -> None:

    if not skip_stereo:
        camera = 'camA'
    
    # make temp directory if it doesnt exist
    if not temp.exists():
        temp.mkdir()
        
    # read summary, images and msgs synced files
    with open(summary, 'r') as file:
        summ_df = pd.read_csv(file)
    images = parse_dir(images_path, metadata, camera, task='inference')
    images = sorted(images, key=lambda path: int(re.search(r'\d{19}', path).group()))
    msgs_synceds = parse_dir(images_path, metadata, task='sync')
    if len(msgs_synceds) == 0:
        print('No msgs_synced files within parent directory')
        return None
    synced = pd.concat(msgs_synceds, ignore_index=True)
    del msgs_synceds; gc.collect() # clean up memory
    update_progress(save, 'configs')

    # intialize stereo params
    if not skip_stereo:
        
         # get image size
        example_img = cv2.imread(images[0])
        ex_w, ex_h = example_img.shape[1], example_img.shape[0]
        
        camA_params = get_camera_params(metadata / f'{camera_stereo[0]}.pbtxt')
        camB_params = get_camera_params(metadata / f'{camera_stereo[1]}.pbtxt')
        
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
                cameraMatrix1=camB_params['camera'], distCoeffs1=camB_params['distortion'],
                cameraMatrix2=camA_params['camera'], distCoeffs2=camA_params['distortion'],
                imageSize=(ex_w, ex_h), R=camB_params['rectification'], T=np.array([8.5*16,0,0]))
        
        leftX, leftY = cv2.initUndistortRectifyMap(
                camB_params['camera'], camB_params['distortion'], R1,
                P1, (ex_w, ex_h), cv2.CV_32FC1
            )
        
        rightX, rightY = cv2.initUndistortRectifyMap(
                camA_params['camera'], camA_params['distortion'], R2,
                P2, (ex_w, ex_h), cv2.CV_32FC1
            )
        
        # initialize stereo wrapper
        wrapper = StereoWrapper(leftX, leftY, rightX, rightY)
        del leftX; del leftY; del rightX; del rightY; gc.collect() # clean up memory

    # initialize models
    model = YOLO(model_path)
    print(f'Loading model: {model_path}')
    update_progress(save, 'models')

    # extract timestamps
    if emerging:
        timestamps = pd.unique(summ_df['Timestamp'])
    else:
        timestamps = np.array([int(path.split('-')[-1].split('.')[0]) for path in images])
        
    # split timestamps into chunks
    parts = int(0.05 * len(timestamps)) # take 5% of the timestamps
    if parts == 0:
        parts = 1
    print(f"Splitting dataset into {parts} parts")
    num_parts = len(timestamps) // parts
    chunks = list(split_into_chunks(timestamps, num_parts))
    update_progress(save, 'images')
    
    # initiate from last chunk
    last_processed_idx = -1  # Default if no files have been processed
    
    for file in temp.iterdir():
        if file.suffix == '.pkl':
            idx = int(file.stem) 
            last_processed_idx = max(last_processed_idx, idx)
    start_idx = last_processed_idx + 1
    
    # extraction loop
    dfs = []
    cols_to_avg = ['Plant ID', 'Height (cm)', 'Area (cm2)', f'{trait.capitalize()} Count']
    for idx, chunk in tqdm(enumerate(chunks[start_idx:], start=start_idx), desc="Processing chunk sets", total=len(chunks) - start_idx):
        
        # Calculate progress percentage
        progress_percentage = ((idx + 1) / len(chunks)) * 100
        mapped_progress = 5 + (progress_percentage * (95 - 5) / 100)
        with open(save.parent / 'extract_progress.txt', 'w') as file:
            file.write(f'{int(round(mapped_progress))}\n')

        # run stereo function
        if skip_stereo:
            for c in chunk:
                # interpolate and process disparity maps (check create stereo for process)
                disparity_args = [images_path, camera, synced, images, c]
                outs = np.array(process_disparity(disparity_args))
                outs_chunk.append(outs)
        else:
            chunk = chunk[chunk != '']
            if any(isinstance(i, np.ndarray) for i in chunk):
                chunk = [int(i) for sublist in chunk for i in (sublist if isinstance(sublist, np.ndarray) else [sublist])]
            else:
                chunk = [int(i) for i in chunk]
            outs_chunk = []
            for c in chunk:
                stereo_args = [synced, images, c, Q, camera_stereo, wrapper]
                outs = np.array(create_stereo(stereo_args))
                outs_chunk.append(outs)
        indices_to_remove = [i for i, arr in enumerate(outs_chunk) 
                            if isinstance(arr, int) or (isinstance(arr, np.ndarray) and arr.ndim == 0)]
        for index in sorted(indices_to_remove, reverse=True):
            del outs_chunk[index]
            del chunk[index]
        del outs; del stereo_args; gc.collect() # clean up memory
        if len(outs_chunk) == 0:
            continue

        # create tif data
        tif_args = list(zip(
            outs_chunk,
            [images]*len(chunk),
            chunk,
            [camera]*len(chunk),
            [skip_stereo]*len(chunk)
        ))
        tiffouts = process_map(tiffout, tif_args, max_workers=num_workers, chunksize=1, disable=True)
        tif_arrs = [item[0] for item in tiffouts]
        geos = [item[1] for item in tiffouts]
        widths = [item[2] for item in tiffouts]
        heights = [item[3] for item in tiffouts]
        del tif_args; del tiffouts; gc.collect() # clean up memory

        # calculate plant dimensions
        extract_args = list(zip(
                outs_chunk,
                tif_arrs,
                geos,
                widths,
                heights,
                chunk,
                [summ_df]*len(chunk),
                [emerging]*len(chunk),
                [plotmap]*len(chunk),
                [trait]*len(chunk),
                [skip_stereo]*len(chunk)
            ))
        extract_outs = list(process_map(extract, extract_args, max_workers=num_workers, chunksize=1, disable=True))
        del extract_args; del outs_chunk; del tif_arrs; gc.collect() # clean up memory

        # count plant traits
        summ_dfs = []
        extract_batches = split_into_batches(extract_outs, batch_size)
        unpacked_batches = [unpack_batch(batch) for batch in extract_batches]
        del extract_outs; del extract_batches; gc.collect() # clean up memory
        for batch_num, (images_batch, summs_batch, indices_batch, bboxes_batch) in enumerate(unpacked_batches):
            
                df = predict(images_batch, summs_batch, indices_batch, bboxes_batch, model, trait, skip_stereo)
                summ_dfs = summ_dfs + df
        del unpacked_batches; del images_batch; del summs_batch; gc.collect() # clean up memory

        # take average across different counts
        combined_df = pd.concat(summ_dfs)
        grouped_df = combined_df.groupby('UTM')
        averaged_df = grouped_df[cols_to_avg].mean().reset_index()
        del summ_dfs; del combined_df; del grouped_df; gc.collect() # clean up memory
        
        filename = f'{idx}.pkl'
        averaged_df.to_pickle(temp / filename)
        
    # load in dataframes from disk memory
    for file in os.listdir(temp):
        df = pd.read_pickle(temp / file)
        dfs.append(df)
        os.remove(temp / file)

    # export trait table and geojson
    combined_df = pd.concat(dfs)
    grouped_df = combined_df.groupby('UTM')
    averaged_df = grouped_df[cols_to_avg].mean().reset_index()
    summ_df_final = pd.merge(summ_df, averaged_df, on=['UTM','Plant ID'])
    summ_df_final.dropna(inplace=True)
    summ_df_final[cols_to_avg] = summ_df_final[cols_to_avg].apply(np.ceil)
    update_progress(save, 'filter')
    summ_df_final.to_csv(Path(save).with_suffix('.csv'), index=False)

    # merge with attributes
    trait_col = f'{trait.capitalize()} Count'
    summ_df_final = summ_df_final.drop(['Timestamp','UTM', 'Plant ID'], axis=1)
    format_df = gpd.read_file(plotmap)
    
    # change `row` and `column` to `Bed` and `Tier` in format_df (if not already)
    if 'Bed' not in format_df.columns:
        format_df.rename(columns={'row':'Bed', 'column':'Tier'}, inplace=True)
        
    # change `Group` and `Label` to `population` and `accession` in summ_df (if not already)
    if 'population' not in summ_df_final.columns:
        summ_df_final.rename(columns={'Group':'population', 'Label':'accession'}, inplace=True)
        
    # lower case `Plot` in summ_df (if not already)
    if 'plot' not in summ_df_final.columns:
        summ_df_final.rename(columns={'Plot':'plot'}, inplace=True)
        
    summ_df_final = summ_df_final.merge(format_df, on=['Bed','Tier'], how='right')
    count_df = summ_df_final.groupby(['Bed', 'Tier']).size().reset_index(name='Stand Count')
    summ_df_final = summ_df_final.groupby(['Bed','Tier']).agg({
        'plot': 'first',
        'accession': 'first',
        'population': 'first',
        'Height (cm)': 'mean',
        'Area (cm2)': 'mean',
        trait_col: 'sum',
        'geometry': 'first'
    }).reset_index()
    summ_df_final = summ_df_final.merge(count_df, on=['Bed', 'Tier'])
    summ_df_final.fillna(0, inplace=True)
    summ_df_final = summ_df_final[['Bed','Tier','plot','accession','population', 'Stand Count',
        'Height (cm)','Area (cm2)', trait_col, 'geometry']]
    summ_df_final['Height (cm)'] = summ_df_final['Height (cm)'].astype(float).round(1)
    summ_df_final['Area (cm2)'] = summ_df_final['Area (cm2)'].astype(float).round(1)
    summ_df_final[trait_col] = summ_df_final[trait_col].astype(int)
    summ_df_final['Bed'] = summ_df_final['Bed'].astype(int)
    summ_df_final['Tier'] = summ_df_final['Tier'].astype(int)
    summ_df_final.rename(columns={'Height (cm)':'Average Height (cm)'}, inplace=True)
    summ_df_final.rename(columns={'Area (cm2)':'Average Leaf Area (cm2)'}, inplace=True)
    summ_df_final.rename(columns={trait_col:f'Total {trait_col}'}, inplace=True)
    out_df = gpd.GeoDataFrame(summ_df_final, geometry='geometry')
    
    # check if file already exists, and if so merge with existing file
    # existing file should have the same columns already as out_df
    # merge on existing columns and add new ones
    path_to_geojson = save / geojson_filename
    if path_to_geojson.exists():
        existing_df = gpd.read_file(path_to_geojson)
        existing_df = existing_df.merge(out_df, on=['Bed','Tier','plot','accession','population','Stand Count',
            'Average Height (cm)','Average Leaf Area (cm2)', f'Total {trait_col}', 'geometry'], how='outer')
        out_df = existing_df
    else:
        out_df.to_file(save / geojson_filename, driver='GeoJSON')
        update_progress(save, 'export')

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('--emerging', action='store_true',
        help='If date selected is emerging and used for stand count.')
    ap.add_argument('--summary', type=Path, required=True,
        help='Path to summary file outputted from locate.py')
    ap.add_argument('--images', type=Path, required=True, 
                    help='Path to parent directory containing images.')
    ap.add_argument('--camera', type=str, default='top',
                    help='Camera view to run inference on.')
    ap.add_argument('--camera-stereo', type=str, nargs='+', default=['camA', 'camB'],
                help='Camera views to create stereo images.')
    ap.add_argument('--plotmap', type=Path, required=True,
                    help='Path to field plot attributes.')
    ap.add_argument('--batch-size', type=int, default=16,
                    help='Batch size for running inference.')
    ap.add_argument('--model-path', type=Path, required=True,
                    help='Path to folder containing models.')
    ap.add_argument('--save', type=Path, required=True,
                    help='Path to save outputs.')
    ap.add_argument('--metadata', type=Path, required=True,
                    help='Path to folder containing camera configuration file')
    ap.add_argument('--temp', type=Path, required=True,
                    help='Temporary folder to store pandas dataframe.')
    ap.add_argument('--trait', type=str, required=True,
                    help='Name of trait the user is analyzing.')
    ap.add_argument('--skip-stereo', action='store_true', default=False,
                    help='True if pointclouds already exist.')
    ap.add_argument('--geojson-filename', type=str, required=True,
                    help='Name of the file to save the geojson')

    args = ap.parse_args()
    
    print(f'num. of workers: {num_workers}')

    main(args.emerging, args.summary, args.images, \
            args.camera, args.camera_stereo, args.plotmap, \
                args.batch_size, args.model_path, args.metadata, args.save, \
                    args.temp, args.trait,args.skip_stereo, args.geojson_filename)
