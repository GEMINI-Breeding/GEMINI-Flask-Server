import torch
import glob
import argparse
import os
import cv2
import shapely
import re
import utm
import time
import warnings
import multiprocessing
import random
import geopandas as gpd
import pandas as pd
import numpy as np

from typing import List
from cv2 import cuda
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from shapely.geometry import Point
from tqdm.contrib.concurrent import process_map

warnings.filterwarnings('ignore')

# globals
device = torch.device("cpu")
num_workers = int(multiprocessing.cpu_count() / 1.5)
progress = 0

class StereoWrapper:
    """
    This class handles stereo image processing using OpenCV.
    """
    def __init__(self,
                 leftX: np.ndarray, 
                 leftY: np.ndarray,
                 rightX: np.ndarray,
                 rightY: np.ndarray,
                 minDisparity: int = 90,
                 numDisparities: int = 4*16,
                 P1: int = 32,
                 P2: int = 150,
                 uniquenessRatio: int = 5
                 ) -> None:

        self.leftX = leftX
        self.leftY = leftY
        self.rightX = rightX
        self.rightY = rightY

        self.stereo_sgbm = cv2.StereoSGBM_create(
            minDisparity=minDisparity,
            numDisparities=numDisparities,
            blockSize=5,
            P1=P1,
            P2=P2,
            uniquenessRatio=uniquenessRatio
        )
        
    def preprocess(self,
                   left_img: np.ndarray,
                   right_img: np.ndarray,
                   platform: str
                   ) -> tuple:
        """
        Preprocesses images for stereo creation depending on sensing platform.
        Args:
            left_img: left image matrix
            right_img: right image matrix
            platform: sensing platform
        Returns:
            The processed images for both left and right
        """
        if platform == 'rover' or platform == 'Amiga':
            left_img = cv2.flip(left_img, -1)
            right_img = cv2.flip(right_img, -1)
            left_img = cv2.remap(left_img, self.leftX, self.leftY, cv2.INTER_LINEAR)
            right_img = cv2.remap(right_img, self.rightX, self.rightY, cv2.INTER_LINEAR)
        return left_img, right_img

    def compute_disparity(self,
                          left_img: np.ndarray,
                          right_img: np.ndarray,
                          platform: str = 'Amiga'
                          ) -> np.ndarray:
        """
        Computes the disparity map using the named algorithm and platform.
        Args:
            left_img: the numpy image matrix for the left camera
            right_img: the numpy image matrix for the right camera
            platform: sensing platform used to collect data
        Returns:
            The disparity map
        """
        left_img, right_img = self.preprocess(left_img, right_img, platform)
        depth = self.stereo_sgbm.compute(left_img, right_img)
        return depth
        

def filter(
    gdf: gpd,
    plotmap: Path,
    save: Path
) -> gpd:

    # apply nms to coordinates
    print('Applying NMS...')
    boxes = gdf[['x1','y1','x2','y2']].values.tolist()
    scores = gdf[['score']].values.flatten()
    timestamps = gdf[['Timestamp']].values.flatten() 
    nms_args = list(split_boxes(boxes, scores, timestamps, 1000))
    nms_outs = list(process_map(nms, nms_args, max_workers=num_workers))
    filt_boxes = [item for sublist in [l[0] for l in nms_outs] for item in sublist]
    filt_timestamps = [item for sublist in [l[2] for l in nms_outs] for item in sublist]
    filt_coords = pd.DataFrame(filt_boxes, columns=['x1','y1','x2','y2'])
    filt_coords['geometry'] = filt_coords.apply(lambda x: shapely.geometry.box(x[2], x[3], x[0], x[1]), axis=1)
    gdf = gpd.GeoDataFrame(filt_coords)
    gdf = gdf.set_crs('EPSG:32610')

    # prepare data for parallel processing
    file = open(plotmap)
    plot_map = gpd.read_file(file)
    plot_map = plot_map.to_crs(plot_map.estimate_utm_crs())
    tasks = [(i, filt_coords.iloc[i], plot_map, filt_timestamps[i]) for i in range(len(filt_coords))]

    # process data in parallel
    print('Assigning plots...')
    results = process_map(match_plot_boundary, tasks, max_workers=None)

    # convert results to DataFrame
    columns = ['Index', 'Timestamp', 'Bed', 'Tier', 'UTM']
    results_df = pd.DataFrame(results, columns=columns).sort_values('Index').reset_index(drop=True)

    # drop the 'Index' column as it is no longer needed
    results_df = results_df.drop(columns=['Index'])
    gdf.to_file(save / "locate.geojson", driver='GeoJSON')

    return results_df

def conversion(
    images_path: Path,
    preds_df: pd,
    msgs_synceds: List[str],
    metadata: Path,
    camera: str,
    camera_stereo: List[str],
    images: List[str],
    save: Path,
    skip_stereo: bool
) -> gpd:
    """
    Create stereo pairs and converts bounding boxes into utm coordinates
    """
    print('Running conversion...')
    
    # check if file exists
    if len(msgs_synceds) == 0:
        print('No msgs_synced files within parent directory')
        return None
    else:
        synced = pd.concat(msgs_synceds, ignore_index=True)
        
    # prepocess boxes
    preds_df['bbox_np'] = preds_df['bbox'].apply(lambda x: np.array(x))
    preds_df.drop(preds_df[preds_df['label'] == 'weed'].index, inplace=True)
    
    # collect timestamps
    preds_df['timestamp'] = preds_df['filename'].apply(lambda x: int(x.split('-')[1][:-4]))
    g = preds_df.groupby('timestamp')
    ts_dfs = {}
    for df in g:
        ts = df[0]
        ts_dfs[ts] = df[1]
    timestamps = pd.unique(preds_df['timestamp'])
    
    # split timestamps into chunks
    parts = int(0.05 * len(timestamps)) # take 5% of the timestamps
    if parts == 0:
        parts = 1
    print(f"Splitting dataset into {parts} parts")
    num_parts = len(timestamps) // parts
    chunks = list(split_into_chunks(timestamps, num_parts))
    
    if not skip_stereo:
        
        # create stereo pairs
        camA_params = get_camera_params(metadata / f'{camera_stereo[0]}.pbtxt')
        camB_params = get_camera_params(metadata / f'{camera_stereo[1]}.pbtxt')
        
        # get image size
        example_img = cv2.imread(images[0])
        ex_w, ex_h = example_img.shape[1], example_img.shape[0]
        
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify( ## NOTE: these values could change
                cameraMatrix1=camB_params['camera'], distCoeffs1=camB_params['distortion'],
                cameraMatrix2=camA_params['camera'], distCoeffs2=camA_params['distortion'],
                imageSize=(ex_w, ex_h), R=camB_params['rectification'], T=np.array([8.5*16,0,0]))
        
        leftX, leftY = cv2.initUndistortRectifyMap( ## NOTE: these values could change
                camB_params['camera'], camB_params['distortion'], R1,
                P1, (ex_w, ex_h), cv2.CV_32FC1
            )
        
        rightX, rightY = cv2.initUndistortRectifyMap( ## NOTE: these values could change
                camA_params['camera'], camA_params['distortion'], R2,
                P2, (ex_w, ex_h), cv2.CV_32FC1
            )
        wrapper = StereoWrapper(leftX, leftY, rightX, rightY)
        
    # run create stereo function
    all_gdfs = []
    pmarker = len(chunks) // 2
    for idx, chunk in tqdm(enumerate(chunks), desc="Processing chunk sets", total=len(chunks)):
        outs_chunk = []
        if skip_stereo:
            for c in chunk:
                # interpolate and process disparity maps (check create stereo for process)
                disparity_args = [images_path, camera, synced, images, c]
                outs = np.array(process_disparity(disparity_args))
                outs_chunk.append(outs)
        else:
            for c in chunk:
                stereo_args = [synced, images, c, Q, camera_stereo, wrapper]
                outs = np.array(create_stereo(stereo_args))
                outs_chunk.append(outs)

        # convert annotated boxes to UTM coordinates
        box_utm = []
        orig_pixels = []
        for s in range(len((outs_chunk))):
            df = ts_dfs[chunk[s]]
            df = df.reset_index(drop=True)
            if type(outs_chunk[s]) != int:
                enens = boxes_to_utm(outs_chunk[s], df)
                box_utm.append(enens)
                orig_pixels.append(df)
            
        # aggregate all boxes into a single geopandas dataframe
        box_utm_dfs = [pd.DataFrame(x) for x in box_utm]
        box_utm_df = pd.concat(box_utm_dfs, axis=0, ignore_index=True)
        if len(box_utm_df) == 0:
            continue
        box_utm_df.columns = ['Timestamp','x1','y1','x2','y2','score']
        box_utm_df = box_utm_df.dropna()
        box_utm_df['geometry'] = box_utm_df.apply(lambda x: shapely.geometry.box(x[3], x[4], x[1], x[2]), axis=1)
        gdf = gpd.GeoDataFrame(box_utm_df)
        gdf = gdf.set_crs('EPSG:32610')

        # append to the list (Optional, only if you want to aggregate all data in the end)
        all_gdfs.append(gdf)
        if pmarker == idx:
            update_progress(save, 'c1') 

    # aggregate all results and save in a single geojson file
    final_gdf = pd.concat(all_gdfs, axis=0, ignore_index=True)
    
    update_progress(save, 'c2')
    return final_gdf

def inference(
    model: YOLO,
    images: List[str],
    batch_size: int,
    save: Path,
    iou: float
) -> pd:
    """
    Runs inference on images and exports results into a dataframe.
    
    Args:

        model (YOLO): model imported using ultralytics YOLO module
        images (list[str]): list containing paths to images
        batch_size (int): size of batch to input for inference
        image_size (int): compression for model input
        iou (float): intersection over union for nms
    """
    print('Running inference...')
    
    # split images into batches
    img_batches = [images[i:i+batch_size] for i in range(0,len(images), batch_size)]
    
    # inference
    img_dfs = []
    pmarker = len(img_batches) // 2
    for idx, batch in enumerate(tqdm(img_batches), 1):
        try:
            # run prediction
            results = model.predict(batch, iou=iou, device=device, verbose=False)
            
            # parse results
            boxes = [results[i].boxes.xywh.cpu().numpy() for i in range(len(results))]
            labels = [results[i].boxes.cls.cpu().numpy() for i in range(len(results))]
            scores = [results[i].boxes.conf.cpu().numpy() for i in range(len(results))]
            filenames = [[os.path.basename(results[i].path)]*len(boxes[i]) for i in range(len(results))]
            
            # store data into dataframe
            batch_df = pd.DataFrame({
                'filename': np.concatenate(filenames),
                'bbox': np.concatenate(boxes).tolist(),
                'labels': np.concatenate(labels),
                'score': np.concatenate(scores)
            })
            img_dfs.append(batch_df)
            
            if idx == pmarker:
                update_progress(save, 'i1')

        except Exception as e:
            print(f'Could not complete prediction: {e}')
            print(f'Possible corrupt data: {batch}')
    
    # concatenate all batch dataframes into a single dataframe
    preds_df = pd.concat(img_dfs, axis=0, ignore_index=True)
    
    # convert int label to str and remove rows containing weeds
    type_dict = {0:'plant',
                1:'weed'}
    preds_df['label_str'] = preds_df['labels'].apply(lambda x: type_dict[int(x)])
    
    # formate and export df
    preds_df_final = preds_df[['filename','label_str','bbox','score']]
    preds_df_final.columns = ['filename','label','bbox','score']
                
    update_progress(save, 'i2')
    return preds_df_final

def match_plot_boundary(args):

    idx, coord, plot_map, timestamp = args
    x, y = coord['x2'], coord['y1']
    point = Point(x, y)

    for plot in plot_map.itertuples():
        if plot.geometry.contains(point):
            return idx, timestamp, plot.column, plot.row, coord[['x1', 'y1', 'x2', 'y2']].tolist()

    return idx, timestamp, None, None, coord[['x1', 'y1', 'x2', 'y2']].tolist()

def nms(
    args: list
) -> list:
    """
    Runs non-maximum suppresion given conf scores and IOU threshold.
    """
    boxes, scores, timestamps = args
    thresh = 0.7

    ious = np.array([])
    
    # If no bounding boxes, return empty list
    if len(boxes) == 0:
        return [], [], []

    # Bounding boxes
    boxes = np.array(boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(scores).flatten()
    
    # Picked bounding boxes
    picked_boxes = []
    picked_score = []
    picked_timestamps = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:

        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(boxes[index])
        picked_score.append(score[index])
        picked_timestamps.append(timestamps[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)
        ious = np.append(ious, ratio)

        left = np.where(ratio < thresh)
        order = order[left]

    return picked_boxes, picked_score, picked_timestamps, ious

def boxes_to_utm(
    out3d: np.ndarray, 
    dets: pd
) -> list:
    """
    Convert bboxes into utm coordinates using outputs from stereo function
    
    Args:
        out3d (np.ndarray): list of outputs from stereo pairing
        dets (pd): dataframe containing detections
    """
    ##NOTE: image parameters could definitely change here
    
    # If there was a problem with the 3D stereo
    if type(out3d) == int:
        return []
    
    # Since the GeoTiffs use this method of spacing out image pixels based on
    # min and max location and number of pixels, we should use the same thing
    # for the boxes so they align and to limit artifacts caused by the stereo 
    # reconstruction process.
    image_size = out3d.shape
    if len(image_size) < 2:
        return []
    nx = image_size[1]
    ny = image_size[0]
    xmin, ymin, xmax, ymax = np.array([np.min(out3d[:,:,0]), np.min(out3d[:,:,1]), 
                                        np.max(out3d[:,:,0]), np.max(out3d[:,:,1])])/100

    xband = np.linspace(xmin, xmax, nx)
    yband = np.linspace(ymax, ymin, ny)

    xy = np.meshgrid(xband,yband)
    xy = np.stack(xy, -1)

    # Now that we have a regular xy grid, we can query it
    # at the locations of the bounding boxes
    enens = []
    for r in range(len(dets)):
        box = dets.loc[r, 'bbox_np'].astype(int)
        score = dets.loc[r, 'score']
        
        if (box[1] < 512) | (box[1]+box[3] > (1024+512)): #minimize box overlap
            continue
        
        box_shift = 90 # 90 if testing with camera A annots
        
        e1 = xy[box[1], box[0]-box_shift, 0]
        n1 = xy[box[1], box[0]-box_shift, 1]
        
        if (box[0]-90+box[2] > 2591) and ((box[1]+box[3]) > 2047):
            e2 = xy[2047, 2592, 0]
            n2 = xy[2047, 2592, 1]
        elif (box[0]-box_shift+box[2] > 2591) and ((box[1]+box[3]) < 2047):
            e2 = xy[(box[1]+box[3]), 2591, 0]
            n2 = xy[(box[1]+box[3]), 2591, 1]
        elif (box[0]-box_shift+box[2] < 2591) and ((box[1]+box[3]) > 2047):
            e2 = xy[2047, (box[0]-box_shift+box[2]), 0]
            n2 = xy[2047, (box[0]-box_shift+box[2]), 1]
        else:        
            e2 = xy[(box[1]+box[3]), (box[0]-box_shift+box[2]), 0]
            n2 = xy[(box[1]+box[3]), (box[0]-box_shift+box[2]), 1]
        
        enen = [dets.loc[r, 'timestamp'], e1, n1, e2, n2, score]
        enens.append(enen)
    
    return enens

def process_disparity(args):
    
    images_path, camera, synced, images, ts = args
    
    # pointclouds are stored in a folder called Disparity
    disparity_path = images_path.replace("Images", "Disparity")
    
    # get row that matches timestamp
    sync_row = synced.loc[synced[f'/{camera}/rgb'] == ts, :]
    heading = sync_row['direction'].values[0]
    
    try:
        # load disparity map and rgb image
        rgb_file = sync_row[f'/{camera}/rgb_file'].values[0]
        disparity_file = sync_row[f'/{camera}/disparity_file'].values[0]
        rgb_im = cv2.imread(os.path.join(images_path, camera, rgb_file), cv2.IMREAD_COLOR)
        out3d = np.load(os.path.join(disparity_path, camera, disparity_file))
        
        # grab location
        lon = sync_row['lon_interp_adj'].values[0]
        lat = sync_row['lat_interp_adj'].values[0]
        e, n, _, _ = utm.from_latlon(lat, lon)
        
        # meters to cm
        e *= 100
        n *= 100
        
        # process out3d
        out3d_np = out3d.astype(np.float64)
        out3d = cv2.UMat(out3d_np)
        multiplier_np = np.ones_like(out3d.get(), dtype=np.float64)
        multiplier_np[:,:,0] = -1
        multiplier = cv2.UMat(multiplier_np)
        out3d = cv2.multiply(out3d, multiplier)
        
        # convert to real coordinates
        if heading == 'North':
            out3d = cv2.flip(out3d, 0)
            out3d = cv2.flip(out3d, 1)
            left_im = cv2.flip(left_im, 0)
            left_im = cv2.flip(left_im, 1)
            out3d = cv2.multiply(out3d, multiplier)
        if heading == 'East':
            # Rotate 90 degrees clockwise
            out3d = cv2.rotate(out3d, cv2.ROTATE_90_CLOCKWISE)
            left_im = cv2.rotate(left_im, cv2.ROTATE_90_CLOCKWISE)
        if heading == 'West':
            # Rotate 90 degrees counterclockwise
            out3d = cv2.rotate(out3d, cv2.ROTATE_90_COUNTERCLOCKWISE)
            left_im = cv2.rotate(left_im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            e -= 0.0 # shift camera over x cm to align with other heading
            
        add_matrix_np = np.ones_like(out3d.get(), dtype=np.float64)
        add_matrix_np[:, :, 0] *= e
        add_matrix_np[:, :, 1] *= n
        add_matrix_umat = cv2.UMat(add_matrix_np)
        out3d = cv2.add(out3d, add_matrix_umat)
        
    except Exception as e:
        print(f'Failed to read cam sync files: {e}')
        return 0
    
    return out3d.get()

def create_stereo(
    args: tuple
) -> list:
    """
    Create stereo images and stores outputs into list
    
    Args:
        args (list): list of parameters to create stereo pairs
    """
    
    synced, images, ts, Q, camera_stereo, wrapper = args
    camA, camB = camera_stereo

    # get row that matches timestamp
    # synced[f'{camA}_time'] = synced[f'{camA}_time'].fillna(0).astype(np.int64)
    synced_timestamps = synced[f'{camA}_time'].fillna(0).to_numpy().astype(np.int64)
    sync_row = synced.loc[synced_timestamps == np.int64(ts), :]
    if len(sync_row['direction'].values) == 0:
        return 0
    heading = sync_row['direction'].values[0]
    
    # try to load in message based on file name in msgs synced
    try:
        
        file = sync_row[f'{camA}_file'].values[0]
        roots = [s for s in images if file.lower() in s.lower()]
        if len(roots) > 1:
            print(f'Multiple files of the same timestamp is found: {roots}')
            return 0
        elif len(roots) == 0:
            print(f'Timestamp not found: {file}')
            return 0
        root = Path(os.path.dirname(roots[0]))
        right_im = cv2.imread(str(root / sync_row[f'{camA}_file'].values[0]), cv2.IMREAD_COLOR)
        left_im = cv2.imread(str(root / sync_row[f'{camB}_file'].values[0]), cv2.IMREAD_COLOR)
        
        # if either right_im or left_im is empty, return 0
        if (right_im is None) or (left_im is None):
            return 0
        
    except Exception as e:
        
        print(f'Failed to read cam sync files: {e}')
        return 0
    
    # Grab location
    lon = sync_row['lon_interp_adj'].values[0]
    lat = sync_row['lat_interp_adj'].values[0]
    e, n, _, _ = utm.from_latlon(lat, lon)
    
    # Meters to cm
    e *= 100
    n *= 100 
    
    # Create stereo
    depth = wrapper.compute_disparity(left_im, right_im)
    
    # Create 3D
    out3d = cv2.reprojectImageTo3D(depth, Q)
    out3d_np = out3d.astype(np.float64)
    out3d = cv2.UMat(out3d_np)
    multiplier_np = np.ones_like(out3d.get(), dtype=np.float64)
    multiplier_np[:,:,0] = -1
    multiplier = cv2.UMat(multiplier_np)
    out3d = cv2.multiply(out3d, multiplier)

    # Convert to real coordinates
    if heading == 'North':
        out3d = cv2.flip(out3d, 0)
        out3d = cv2.flip(out3d, 1)
        left_im = cv2.flip(left_im, 0)
        left_im = cv2.flip(left_im, 1)
        out3d = cv2.multiply(out3d, multiplier)
    else:
        e -= 8.5 # shift camera over 8.5 cm to align with other heading
    
    add_matrix_np = np.ones_like(out3d.get(), dtype=np.float64)
    add_matrix_np[:, :, 0] *= e
    add_matrix_np[:, :, 1] *= n
    add_matrix_umat = cv2.UMat(add_matrix_np)
    out3d = cv2.add(out3d, add_matrix_umat)
    
    return out3d.get()

def parse_dir(
    images_path: Path,
    metadata: Path,
    camera: str = None,
    task: str = None,
) -> List[str]:
    """
    Obtain all images contained within parent directory that end with .jpg
    
    Args:
        images_path (Path): path to parent directory
        camera (str): camera view to use
    """
    if task == 'inference':
        # Create a list of image extensions you want to search for
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.gif']

        # Initialize an empty list to hold the paths of the found images
        images = []

        # Loop through each image extension and use glob to search recursively
        for ext in image_extensions:
            search_pattern = str(images_path / '**' / ext)
            try:
                images.extend(glob.glob(search_pattern, recursive=True))
            except Exception as e:
                print(f'Possible incompatible directory: {e}')
        
        return images
    elif task == 'sync':
        # Create a search pattern to find msgs_synced files
        search_pattern = str(metadata / '**' / 'msgs_synced.csv')
        
        # Use glob to get all image paths matching the search pattern
        try:
            msgs_synceds = glob.glob(search_pattern, recursive=True)
            msgs_synceds = [pd.read_csv(msg) for msg in msgs_synceds]
            
            # check in image location if it doesnt exist in metadata
            if len(msgs_synceds) == 0:
                search_pattern = str(images_path / '**' / 'msgs_synced.csv')
                msgs_synceds = glob.glob(search_pattern, recursive=True)
                msgs_synceds = [pd.read_csv(msg) for msg in msgs_synceds]
        except Exception as e:
            print(f'Possible incompatible directory: {e}')
            
        return msgs_synceds
    else:
        return []
    
def get_camera_params(
    pbtxt: Path
) -> dict:
    """
    Get necessasry camera intrinsics and extrinsics for pbtxt file
    
    Args:
        pbtxt (Path): Path to camera configuration pbtxt file
    """
    
    with open(pbtxt) as f:
        t = f.read()
    lines = [x.lstrip() for x in t.split('\n')]

    # Locations of params beginnings
    param_idxs = [s for s,x in enumerate(lines) if 'params' in x]
    param_names = [lines[x].split('_')[0] for x in param_idxs]
    col_nums = [int(lines[x+1].split(': ')[1]) for x in param_idxs]

    camera_params = {}
    for s in range(len(param_idxs)):
        vals = []
        stillval = True
        i = param_idxs[s]+2
        while stillval == True:
            l = lines[i]
            if 'val' in l:
                vals.append(float(l.split(': ')[1]))
                i+=1
            else:
                stillval = False
        if col_nums[s] == 1:
            camera_params[param_names[s]] = np.array(vals)
        else:
            camera_params[param_names[s]] = np.array(vals).reshape(-1,col_nums[s])
            
    return camera_params

def split_into_chunks(
    timestamps: list,
    num_parts: int
):
    """
    Split timestamps list into chunks for more reliable processing
    """

    for i in range(0, len(timestamps), num_parts):
        yield timestamps[i:i + num_parts]
        
def split_boxes(
    boxes: list, 
    scores: list, 
    timestamps: list, 
    batch_size: int
):
     """Split boxes and scores into batches."""
     for i in range(0, len(boxes), batch_size):
         yield boxes[i:i+batch_size], scores[i:i+batch_size], timestamps[i:i+batch_size]
         
def update_progress(
    save: Path,
    task: str
) -> int:
    """Update progress based on task accomplished."""
    scores = {'configs': 1,'models': 2,'images': 10, \
                'i1': 20, 'i2': 40, 'c1': 60, 'c2': 80, \
                        'filter': 90,'export': 100}
    progress = scores[task]
    
    # write in text file (make text file if it doesnt exist)
    with open(save/'locate_progress.txt', 'w') as file:
        file.write(str(progress))

def main(
    images_path: Path,
    camera: str,
    camera_stereo: List[str],
    batch_size: int,
    metadata: Path,
    model: Path,
    save: Path,
    plotmap: Path,
    skip_stereo: bool,
    accelerate: bool
) -> None:
    
    if not skip_stereo:
        camera = 'camA'
    
    # get configuration files
    update_progress(save, 'configs')
    
    # load plant-det model
    start = time.time()
    model = YOLO(model)
    print(f'Sucessfully loaded model')
    update_progress(save, 'models')
    
    # parse through parent directory for images
    images = parse_dir(images_path, metadata, camera, task='inference')
    images = sorted(images, key=lambda path: int(re.search(r'\d{19}', path).group()))
    if accelerate:
        images = images[::3]
    update_progress(save, 'images')

    # run inference on dataset and export dataframe    
    start = time.time()
    preds_df = inference(model, images, batch_size, save, iou=0.5)
    end = time.time()
    preds_df.to_csv(save / "predictions.csv", index=False)
    print(f'elapsed time for inference: {end-start}')
    
    # parse through parent directory for msgs_synced file(s)
    msgs_synceds = parse_dir(images_path, metadata, task='sync')
    
    # convert predictions to geo coordinates
    start = time.time()
    gdf = conversion(images_path, preds_df, msgs_synceds, metadata, \
                            camera, camera_stereo, images, save, skip_stereo)
    end = time.time()
    print(f'elapsed time for conversion: {end-start}')

    # apply nms to resulting coordinates
    start = time.time()
    filt_df = filter(gdf, plotmap, save)
    end = time.time()
    print(f'elapsed time for filter: {end-start}')
    update_progress(save, 'filter')
    
    # postprocess dataframe
    filt_df['Timestamp'] = filt_df['Timestamp'].astype(str) # to maintain precision
    sort_df = filt_df.sort_values(by=['Bed','Tier'], ascending=[True, True])
    clean_df = sort_df.dropna()
    clean_df['Plant ID'] = np.arange(1, len(clean_df)+1)
    clean_df.to_csv(save / "locate.csv", index=False)
    update_progress(save, 'export')

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument('--images', type=Path, required=True, 
                    help='Path to parent directory containing images.')
    ap.add_argument('--camera', type=str, default='top',
                    help='Camera view to run inference on.')
    ap.add_argument('--camera-stereo', type=str, nargs='+', default=['camA', 'camB'],
                    help='Camera views to create stereo images.')
    ap.add_argument('--batch-size', type=int, default=16,
                    help='Batch size for running inference.')
    ap.add_argument('--model', type=Path, required=True,
                    help='Path to model weights.')
    ap.add_argument('--plotmap', type=Path, required=True,
                    help='Path to field plot attributes.')
    ap.add_argument('--save', type=Path, required=True,
                    help='Path to save outputs.') 
    ap.add_argument('--metadata', type=Path, required=True,
                    help='Path to camera configuration file')
    ap.add_argument('--skip-stereo', action='store_true', default=False,
                    help='True if pointclouds already exist.')
    ap.add_argument('--accelerate', action='store_true', default=False,
                    help='True if using high frame rate cameras.')
    args = ap.parse_args()
    
    print(f'num. of workers: {num_workers}')
    
    main(args.images, args.camera, args.camera_stereo, args.batch_size, \
            args.metadata, args.model, args.save, args.plotmap, args.skip_stereo, \
                args.accelerate)
    
    # example: python locate.py --images /home/gemini/mnt/d/GEMINI-App-Data-DEMO/Raw/2022/Subset/Davis/Cowpea/2022-06-20/T4/RGB/Images
    # --model /home/gemini/mnt/d/GEMINI-App-Data-DEMO/Intermediate/2022/Subset/Davis/Cowpea/Training/T4/RGB Plant Detection/Plant-NTA1T6/weights/last.pt
    # --plotmap /home/gemini/mnt/d/GEMINI-App-Data-DEMO/Intermediate/2022/Subset/Davis/Cowpea/Plot-Boundary-WGS84.geojson
    # --save /home/gemini/mnt/d/Temporary/locate_test
    # --metadata /home/gemini/mnt/d/GEMINI-App-Data-DEMO/Raw/2022/Subset/Davis/Cowpea/2022-06-20/T4/RGB/Metadata