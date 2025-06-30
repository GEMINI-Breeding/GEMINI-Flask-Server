import argparse
import cv2
import os
import json
import torch
import kornia as K
import numpy as np
import pandas as pd
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path
from datetime import datetime, timezone
from scipy.spatial import KDTree
from scipy.interpolate import interp1d
from google.protobuf import json_format
from kornia_rs import ImageDecoder
from kornia.core import tensor
from tqdm import tqdm
from typing import List, Dict

from farm_ng.oak import oak_pb2
from farm_ng.gps import gps_pb2
from farm_ng.core.events_file_reader import build_events_dict
from farm_ng.core.events_file_reader import EventsFileReader
from farm_ng.core.events_file_reader import EventLogPosition

import warnings
warnings.filterwarnings("ignore")

# camera positions
CAMERA_POSITIONS = {'oak0': 'top', 'oak1': 'left', 'oak2': 'right'}

# image and gps topics
IMAGE_TYPES = ['rgb']
GPS_TYPES = ['pvt','relposned']
CALIBRATION = ['calibration']
TYPES = IMAGE_TYPES + GPS_TYPES + CALIBRATION

# gps data to analyze
GPS_PVT = ['stamp','gps_time','longitude','latitude','altitude','heading_motion',
            'heading_accuracy','speed_accuracy','horizontal_accuracy','vertical_accuracy',
            'p_dop','height']
GPS_REL = ['stamp','relative_pose_north','relative_pose_east','relative_pose_down',
            'relative_pose_heading','relative_pose_length','rel_pos_valid','rel_heading_valid',
            'accuracy_north','accuracy_east','accuracy_down','accuracy_length','accuracy_heading']

def interpolate_gps(
    gps_dfs: List[np.ndarray],
    image_dfs: List[np.ndarray],
    skip_pointer: int,
    save_path: Path,
    columns
) -> list:
    
    gps_dfs_list = []
    save_path = save_path / 'Metadata'
    
    for key, gps in gps_dfs.items():
        fn = interp1d(
            x=gps[skip_pointer:, 0],
            y=gps[skip_pointer:, 1:],    # now for RELPOSNED this y-array no longer has gps_time
            axis=0, kind='linear', fill_value='extrapolate'
        )
        interpolated = np.array([fn(t) for t in image_dfs[0][skip_pointer:, 0]])
        merged = np.hstack([
            image_dfs[0][skip_pointer:, 0].reshape(-1, 1),
            interpolated
        ])
        new_cols = ['image_timestamp'] + columns[key][1:]

        # save & update
        gps_dfs_list.append(merged)
        gps_dfs_new = pd.DataFrame(merged, columns=new_cols)
        columns[key] = new_cols
        
        # combine column 'image_timestamp' with 'stamp' (make sure no duplicates are present and sort rows by 'stamp')
        gps_dfs_new['stamp'] = gps_dfs_new['image_timestamp']
        gps_dfs_new = gps_dfs_new.drop(columns=['image_timestamp'])
        gps_dfs_new = gps_dfs_new.drop_duplicates(subset=['stamp'])
        gps_dfs_new = gps_dfs_new.sort_values(by=['stamp'])
        gps_dfs_new.reset_index(drop=True, inplace=True)
 
        old = (save_path / f"gps_{key}.csv")
        if key == 'pvt':
            gps_df_old = pd.read_csv(old) if old.exists() else pd.DataFrame(columns=GPS_PVT)
        else:
            gps_df_old = pd.read_csv(old) if old.exists() else pd.DataFrame(columns=GPS_REL)

        combined = pd.concat([gps_df_old.reset_index(drop=True),
                              gps_dfs_new.reset_index(drop=True)],
                             ignore_index=True)
        combined.to_csv(old, index=False)
            
    return gps_dfs_list
        
def process_disparity(
    img: torch.Tensor,
    calibration: dict,
) -> np.ndarray:
    """Process the disparity image.

    Args:
        img (np.ndarray): The disparity image.

    Returns:
        torch.Tensor: The processed disparity image.
    """
    
    # get camera matrix
    intrinsic_data = calibration['cameraData'][2]['intrinsicMatrix']
    fx,fy, cx, cy = intrinsic_data[0], intrinsic_data[4], intrinsic_data[2], intrinsic_data[5]
    camera_matrix = tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    # unpack disparity map
    disparity_t = torch.from_dlpack(img)
    disparity_t = disparity_t[..., 0].float()
    
    # resize disparity map to match rgb image
    disparity_t = F.interpolate(
        disparity_t.unsqueeze(0).unsqueeze(0), size=(1080, 1920), mode='bilinear', align_corners=False\
    )
    disparity_t = disparity_t.squeeze(0).squeeze(0)
    
    # compute depth image from disparity image
    calibration_baseline = 0.075 #m
    calibration_focal = float(camera_matrix[0, 0])
    depth_t = K.geometry.depth.depth_from_disparity(
        disparity_t, baseline=calibration_baseline, focal=calibration_focal
    )
    
    # compute the point cloud from depth image
    points_xyz = K.geometry.depth.depth_to_3d_v2(depth_t, camera_matrix)
    
    return points_xyz.numpy()

def heading_to_direction(heading):
    if heading is not None:
        if (heading > 315 or heading <= 45):
            return 'North'
        elif (heading > 45 and heading <= 135):
            return 'East'
        elif (heading > 135 and heading <= 225):
            return 'South'
        elif (heading > 225 and heading <= 315):
            return 'West'
    else:
        return None

def postprocessing(
    msgs_df: pd.DataFrame, 
    images_cols: List[str]
) -> pd.DataFrame:
    
    # convert timestamps into int64
    msgs_df[images_cols] = msgs_df[images_cols].astype('int64')

    # add columns for file names
    images_cols_new = []
    for col in images_cols:
        new_col = f"{col}_file"
        if 'disparity' in col:
            msgs_df[new_col] = col + '-' + msgs_df[col].astype(str) + '.npy'
        else:
            msgs_df[new_col] = col + '-' + msgs_df[col].astype(str) + '.jpg'
        images_cols_new += [col, new_col]

    # convert heading motion to direction
    if 'heading_motion' in msgs_df.columns:
        msgs_df['direction'] = msgs_df['heading_motion'].apply(heading_to_direction)

    # rename lat/lon columns if they exist
    if 'longitude' in msgs_df.columns:
        msgs_df.rename(columns={'longitude': 'lon'}, inplace=True)
    if 'latitude' in msgs_df.columns:
        msgs_df.rename(columns={'latitude': 'lat'}, inplace=True)

    return msgs_df

# Zhenghao Fei, PAIBL 2020 (edited by Earl Ranario, PAIBL 2025)
def sync_msgs(
    msgs: List[np.array], 
    dt_threshold=None,
    apply_dt_threshold=False
) -> List[np.array]:
    """
    Syncs multiple messages based on their time stamps.
    `msgs` should be a list of numpy arrays, each with timestamps in the first column.
    Synchronization is based on the first message in the list.

    Args:
        msgs (list[np.array]): Messages to sync, timestamps in first column.
        dt_threshold (float, optional): Max allowed time difference to accept a match.
        apply_dt_threshold (bool, optional): If False, disables threshold check. Defaults to True.

    Returns:
        list[np.array]: Synced messages.
    """
    # Ensure reference timestamps are sorted
    ref_msg = msgs[0]
    sort_idx = np.argsort(ref_msg[:, 0])
    msgs[0] = ref_msg[sort_idx]
    msg1_t = msgs[0][:, 0]

    # If needed, estimate dt_threshold based on mean period
    if apply_dt_threshold:
        if dt_threshold is None:
            dt_threshold = np.mean(np.diff(msg1_t))

    # Build KDTree for each other message
    timestamps_kd_list = []
    for msg in msgs[1:]:
        timestamps_kd = KDTree(np.asarray(msg[:, 0]).reshape(-1, 1))
        timestamps_kd_list.append(timestamps_kd)

    # Find index matches within threshold (if enabled)
    msgs_idx_synced = []
    for msg1_idx, t in enumerate(msg1_t):
        msg_idx_list = [msg1_idx]
        dt_valid = True
        for timestamps_kd in timestamps_kd_list:
            dt, msg_idx = timestamps_kd.query([t])
            if apply_dt_threshold and abs(dt) > dt_threshold:
                dt_valid = False
                break
            msg_idx_list.append(msg_idx)

        if dt_valid:
            msgs_idx_synced.append(msg_idx_list)

    # Format output
    msgs_idx_synced = np.asarray(msgs_idx_synced).T
    msgs_synced = []
    for i, msg in enumerate(msgs):
        msg_synced = msg[msgs_idx_synced[i]]
        msgs_synced.append(msg_synced)

    return msgs_synced

def extract_images(
    image_topics: List[str],
    events_dict: Dict[str, List[EventLogPosition]],
    calibrations: Dict[str, dict],
    output_path: Path,
    current_ts: int,
) -> bool:
    """Extracts images as jpg and stores timestamps into a csv file where they are synced based
    on their sequence number.

    ASSUMPTION: GPS is not synced with camera capture.

    Args:

        image_topics (list[str]): Topics that contain image information.
        events_dict (dict[str, list[EventLogPosition]]): All events stored in the binary file containing log info.
        disparity_scale (int): Scale for amplifying disparity color mapping. Default: 1.
        output_path (Path): Path to save images and timestamps.
    """

    print('--- image extraction ---')
    
    # initialize save path
    save_path = output_path / 'Metadata'
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
    
    # convert image topics to camera locations
    image_topics_location = [f"/{CAMERA_POSITIONS[topic.split('/')[1]]}/{topic.split('/')[2]}" \
        for topic in image_topics]
    
    cols = ['sequence_num'] + image_topics_location
    ts_df: pd.DataFrame = pd.DataFrame(columns=cols) 
    
    # define image decoder
    image_decoder = ImageDecoder()

    # loop through each topic
    for topic_name in image_topics:
        
        # initialize camera events and event log
        camera_events: list[EventLogPosition] = events_dict[topic_name]
        event_log: EventLogPosition

        # prepare save path
        camera_name = topic_name.split('/')[1]
        camera_name = CAMERA_POSITIONS[camera_name]
        camera_type = topic_name.split('/')[2]
        topic_name_location = f'/{camera_name}/{camera_type}'
        camera_type = 'Disparity' if camera_type == 'disparity' else 'Images'
        camera_path = output_path / camera_type / camera_name
        if not camera_path.exists():
            camera_path.mkdir(parents=True, exist_ok=True)

        # loop through events write to jpg/npy
        for event_log in tqdm(camera_events):
            # parse the iamge
            sample: oak_pb2.OakFrame = event_log.read_message()

            # decode image
            img = cv2.imdecode(np.frombuffer(sample.image_data, dtype="uint8"), cv2.IMREAD_UNCHANGED)

            # extract image metadata
            sequence_num: int = sample.meta.sequence_num
            timestamp: float = sample.meta.timestamp
            updated_ts: int = int((timestamp*1e6) + current_ts)
            if not sequence_num in ts_df['sequence_num'].values:
                new_row = {col: sequence_num if col == 'sequence_num' else np.nan for col in ts_df.columns}
                ts_df = pd.concat([ts_df, pd.DataFrame([new_row])], ignore_index=True)
            ts_df.loc[ts_df['sequence_num'] == sequence_num, topic_name_location] = updated_ts

            # save image
            if "disparity" in topic_name:
                img = image_decoder.decode(sample.image_data)
                
                if calibrations is None or not camera_name in calibrations:
                    # Just save the raw disparity image if no calibration is available
                    img_name: str = f"disparity-{updated_ts}.npy"
                    np.save(str(camera_path / img_name), img)
                else:
                    points_xyz = process_disparity(img, calibrations[camera_name])
                    img_name: str = f"disparity-{updated_ts}.npy"
                    np.save(str(camera_path / img_name), points_xyz)
            else:
                
                if camera_name == 'top' and calibrations and camera_name in calibrations:
                    calibration = calibrations[camera_name]
                    intrinsic = calibration["cameraData"][2]["intrinsicMatrix"]
                    D = np.array(calibration["cameraData"][2]["distortionCoeff"])

                    # adjust K if your image is downscaled!
                    h, w = img.shape[:2]
                    orig_w = calibration["cameraData"][2]["width"]
                    orig_h = calibration["cameraData"][2]["height"]

                    scale_x = w / orig_w
                    scale_y = h / orig_h

                    K = np.array([
                        [intrinsic[0] * scale_x, 0, intrinsic[2] * scale_x],
                        [0, intrinsic[4] * scale_y, intrinsic[5] * scale_y],
                        [0, 0, 1]
                    ])

                    # undistort using remap
                    R = np.eye(3)
                    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
                    map1, map2 = cv2.initUndistortRectifyMap(K, D, R, new_K, (w, h), cv2.CV_16SC2)
                    img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

                    # CROP to remove black border
                    x, y, w_roi, h_roi = roi
                    img = img[y : y + h_roi, x : x + w_roi]
                
                img_name: str = f"rgb-{updated_ts}.jpg"
                cv2.imwrite(str(camera_path / img_name), img)

    # split dataframe based on columns
    dfs = []
    ts_cols_list = []
    images_report = {}
    unique_camera_ids = {s.split('/')[1] for s in image_topics if s.startswith('/oak')}
    for i in unique_camera_ids:
        i = CAMERA_POSITIONS[i]
        ts_cols = [f'/{i}/rgb',f'/{i}/disparity']
        
        # check for missing topics
        existing_cols = [col for col in ts_cols if col in ts_df.columns]
        missing_cols = [col for col in ts_cols if col not in ts_df.columns]
        
        # report existing and missing columns
        images_report[i] = {
            'existing': existing_cols,
            'missing': missing_cols
        }
        
        if missing_cols:
            print(f"Warning: Skipping missing columns for camera '{i}': {missing_cols}")

        if not existing_cols:
            print(f"Warning: No existing timestamp columns found for camera '{i}'. Skipping this camera.")
            continue  # skip this camera completely

        
        ts_df_split = ts_df[existing_cols]
        ts_df_split = ts_df_split.dropna(subset=existing_cols)
        
         # Check if existing timestamps CSV exists
        if (save_path / f"{i}_timestamps.csv").exists():
            ts_df_existing = pd.read_csv(f"{save_path}/{i}_timestamps.csv")
            ts_df_split = pd.concat([ts_df_existing, ts_df_split], ignore_index=True)

        # Output dataframe as CSV
        ts_df_split.to_csv(f"{save_path}/{i}_timestamps.csv", index=False)
        dfs.append(ts_df_split.to_numpy(dtype='float64'))
        ts_cols_list += existing_cols
        
    return dfs, ts_cols_list, images_report

def extract_gps(
    gps_topics: List[str],
    events_dict: Dict[str, List[EventLogPosition]],
    output_path: Path,
    current_ts: int
) -> bool:
    """Extracts camera extrinsics/intrinsics from calibration event.

    Args:
        gps_topics (list[str]): Topics that contain gps information.
        events_dict (dict[str, list[EventLogPosition]]): All events stored in the binary file containing log info.
        output_path (Path): Path to save images and timestamps.
        current_ts (int): Base timestamp in microseconds.
    """ 

    print('--- gps extraction ---')
    
    df = {}
    gps_cols_list = {}
    gps_metric_summary = {}
    
    # initialize save path
    save_path = output_path / 'Metadata'
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    # loop through each topic
    for topic_name in gps_topics:

        gps_name = topic_name.split('/')[2]
        if gps_name == 'pvt':
            gps_df = pd.read_csv(f"{save_path}/gps_{gps_name}.csv") if (save_path / f"gps_{gps_name}.csv").exists() else pd.DataFrame(columns=GPS_PVT)
        elif gps_name == 'relposned':
            gps_df = pd.read_csv(f"{save_path}/gps_{gps_name}.csv") if (save_path / f"gps_{gps_name}.csv").exists() else pd.DataFrame(columns=GPS_REL)
        else:
            print('Unknown topic name.')
            return False

        gps_events: list[EventLogPosition] = events_dict[topic_name]

        for event_log in tqdm(gps_events):
            if gps_name == 'pvt':
                sample: gps_pb2.GpsFrame = event_log.read_message()
            elif gps_name == 'relposned':
                sample: gps_pb2.RelativePositionFrame = event_log.read_message()
            else:
                print('Unknown protocol message.')
                return False
            
            # Updated timestamp based on current_ts and message timestamp delta
            updated_ts = int(current_ts + (sample.stamp.stamp * 1e6))

            # Create row for GPS data
            if gps_name == 'pvt':
                # --- Convert utc_stamp to epoch time (in microseconds) ---
                utc = sample.utc_stamp

                # Handle potentially negative nano field
                nanos = max(utc.nano, 0)
                dt = datetime(
                    utc.year, utc.month, utc.day,
                    utc.hour, utc.min, utc.sec,
                    nanos // 1000,  # convert nanoseconds to microseconds
                    tzinfo=timezone.utc
                )
                gps_epoch_us = int(dt.timestamp() * 1e6)
                
                new_row = {
                    'stamp': [updated_ts], 'gps_time': [gps_epoch_us],
                    'longitude': [sample.longitude], 'latitude': [sample.latitude],
                    'altitude': [sample.altitude], 'heading_motion': [sample.heading_motion], 
                    'heading_accuracy': [sample.heading_accuracy], 'speed_accuracy': [sample.speed_accuracy], 
                    'horizontal_accuracy': [sample.horizontal_accuracy], 'vetical_accuracy': [sample.vertical_accuracy], 
                    'p_dop': [sample.p_dop], 'height': [sample.height]
                }
            elif gps_name == 'relposned':
                new_row = {
                    'stamp': [updated_ts],
                    'relative_pose_north': [sample.relative_pose_north], 'relative_pose_east': [sample.relative_pose_east],
                    'relative_pose_down': [sample.relative_pose_down], 'relative_pose_heading': [sample.relative_pose_heading],
                    'relative_pose_length': [sample.relative_pose_length], 'rel_pos_valid': [sample.rel_pos_valid],
                    'rel_heading_valid': [sample.rel_heading_valid], 'accuracy_north': [sample.accuracy_north],
                    'accuracy_east': [sample.accuracy_east], 'accuracy_down': [sample.accuracy_down],
                    'accuracy_length': [sample.accuracy_length], 'accuracy_heading': [sample.accuracy_heading]
                }

            new_df = pd.DataFrame(new_row)
            new_df.reset_index(inplace=True, drop=True)
            gps_df.reset_index(inplace=True, drop=True)
            gps_df = pd.concat([gps_df, new_df], ignore_index=True)

        gps_df.replace({'True': 1, 'False': 0}, inplace=True)
        gps_df = gps_df.apply(pd.to_numeric, errors='coerce')
        gps_df.to_csv(f"{save_path}/gps_{gps_name}.csv", index=False)
        df[gps_name] = gps_df.to_numpy(dtype='float64')
        gps_cols_list[gps_name] = gps_df.columns.tolist()
        
        # record gps metric summary (average values)
        if gps_name == 'pvt':
            gps_metric_summary['pvt'] = {
                'avg_heading_accuracy': gps_df['heading_accuracy'].mean(),
                'avg_speed_accuracy': gps_df['speed_accuracy'].mean(),
                'avg_horizontal_accuracy': gps_df['horizontal_accuracy'].mean(),
                'avg_vertical_accuracy': gps_df['vertical_accuracy'].mean(),
            }
        elif gps_name == 'relposned':
            gps_metric_summary['relposned'] = {
                'avg_accuracy_north': gps_df['accuracy_north'].mean(),
                'avg_accuracy_east': gps_df['accuracy_east'].mean(),
                'avg_accuracy_down': gps_df['accuracy_down'].mean(),
                'avg_accuracy_length': gps_df['accuracy_length'].mean(),
                'avg_accuracy_heading': gps_df['accuracy_heading'].mean()
            }

    return df, gps_cols_list, gps_metric_summary

def extract_calibrations(
    calib_topics: List[str],
    events_dict: Dict[str, List[EventLogPosition]],
    output_path: Path
) -> bool:
    """Extracts camera extrinsics/intrinsics from calibration event.

    Args:

        calib_topics (list[str]): Topics that contain image calibration information.
        events_dict (dict[str, list[EventLogPosition]]): All events stored in the binary file containing log info.
        output_path (Path): Path to save images and timestamps.
    """ 

    print('--- calibration extraction ---')
    
    # initialize save path
    save_path = output_path / 'Metadata'
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    # loop through each topic
    calibrations = {}
    for topic_name in calib_topics:

        # prepare save path
        camera_name = topic_name.split('/')[1]

        # initialize calib events and event log
        calib_events: list[EventLogPosition] = events_dict[topic_name]
        event_log: EventLogPosition

        # check event log to extract information
        for event_log in tqdm(calib_events): # *calibration events should remain unchange during the collection

            # read message
            calib_msg = event_log.read_message()
            json_data: dict = json_format.MessageToDict(calib_msg)

            # store as pbtxt file
            camera_name = CAMERA_POSITIONS[camera_name]
            json_name = f'{camera_name}_calibration.json'
            json_path = save_path / json_name
            
            # store data
            calibrations[camera_name] = json_data
            
            # check if json file exists
            if json_path.exists():
                continue
            else:
                with open(json_path, "w") as json_file:
                    json.dump(json_data, json_file, indent=4)

    return calibrations

def extract_binary(file_names, output_path) -> None:
    """Read an events file and extracts relevant information from it.

    Args:

        file_name (Path): The path to the events file.
        output_path (Path): The path to the folder where the converted data will be written.
        disparity_scale (int): Scale for amplifying disparity color mapping. Default: 1.
    """
    # print out file names
    print(f"Extracting {len(file_names)} files.")
    
    # make output directory
    base = 'RGB'
    output_path = output_path / base
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True) # *: this path should not change
    
    # write progress text file
    counter = 0
    skip_pointer = 0
    with open(f"{output_path}/progress.txt", "w") as f:
        f.write("0")
        
    # create a report file
    report_path = output_path / 'report.txt'
    with open(report_path, "w") as f:
        f.write("Report of the conversion process:\n")
        f.write(f"Number of files: {len(file_names)}\n")
        f.write(f"Output path: {output_path}\n")
    
    # extract each bin file
    for file_name in tqdm(file_names):
        
        # write name of binary file to the report
        with open(report_path, "a") as f:
            f.write(f"\n--- File: {file_name} ---\n")
        
        # create the file reader
        reader = EventsFileReader(file_name)
        success: bool = reader.open()
        if not success:
            raise RuntimeError(f"Failed to open events file: {file_name}")

        # get the index of the events file
        events_index = reader.get_index()

        # structure the index as a dictionary of lists of events
        events_dict: dict[str, list[EventLogPosition]] = build_events_dict(events_index)
        all_topics = list(events_dict.keys())
        print(f"All available topics: {sorted(events_dict.keys())}")

        # keep only relevant topics
        topics = [topic for topic in all_topics if any(type_.lower() in topic.lower() for type_ in TYPES)]
    
        # *add loop going through each .bin file and update progress.txt
        # get datetime of recording 
        if len(os.path.basename(file_name).split('_')) < 7:
            raise RuntimeError("'File name is not compatible with this script.")
        date_contents = os.path.basename(file_name).split('_')[:7]
        date_string = '_'.join(date_contents)
        date_format = '%Y_%m_%d_%H_%M_%S_%f'
        date_object = datetime.strptime(date_string, date_format).replace(tzinfo=timezone.utc)
        current_ts  = int(date_object.timestamp() * 1e6)
        
        # extract calibration topics
        calib_topics = [topic for topic in topics if any(type_.lower() in topic.lower() for type_ in CALIBRATION)]
        calibrations: dict[str, dict] = extract_calibrations(calib_topics, events_dict, output_path)
        if len(calibrations) == 0:
            print("Warning: No calibration files found. Skipping point cloud generation from disparity images.")
            calibrations = None
        else:
            # write title to the report file with indent
            with open(report_path, "a") as f:
                f.write("\n    --- Calibration ---\n")

            # for each key in calibration, log into the report file
            for key, value in calibrations.items():
                with open(report_path, "a") as f:
                    f.write(f"      Camera: {key}\n")

        # extract gps topics
        gps_topics = [topic for topic in topics if any(type_.lower() in topic.lower() for type_ in GPS_TYPES)]
        gps_dfs, gps_cols, gps_metric_summary = extract_gps(gps_topics, events_dict, output_path, current_ts)
        if len(gps_dfs) == 0:
            raise RuntimeError("Failed to extract gps event file")
        else:
            # write title to the report file with indent
            with open(report_path, "a") as f:
                f.write("\n    --- GPS ---\n")
                
            # for each key in gps, log into the report file
            for key, value in gps_metric_summary.items():
                with open(report_path, "a") as f:
                    f.write(f"      GPS: {key}\n")
                    for metric, val in value.items():
                        f.write(f"        {metric}: {val}\n")

            # for each key in gps, log into the report file
            for key, value in gps_dfs.items():
                with open(report_path, "a") as f:
                    f.write(f"      GPS: {key}\n")

        # extract image topics
        image_topics = [topic for topic in topics if any(type_.lower() in topic.lower() for type_ in IMAGE_TYPES)]
        image_dfs, images_cols, images_report = extract_images(image_topics, events_dict, calibrations, output_path, current_ts)
        # check each image_dfs if it is empty
        for df_check in image_dfs:
            if len(df_check) == 0:
                print("Warning: No images found for one of the image dataframes.")
                
        if len(image_dfs) == 0:
            raise RuntimeError("Failed to extract image event file")
        else:
            # write title to the report file with indent
            with open(report_path, "a") as f:
                f.write("\n    --- Images ---\n")
                
            # for each key in image, log into the report file
            with open(report_path, "a") as f:
                f.write(f"      Images: {image_topics}\n")
                for key, value in images_report.items():
                    f.write(f"        Camera: {key}\n")
                    f.write(f"          Existing topics: {value['existing']}\n")
                    f.write(f"          Missing topics: {value['missing']}\n")
        
        # *: Interpolate GPS data to query at camera timestamps
        gps_dfs = interpolate_gps(gps_dfs = gps_dfs, image_dfs = image_dfs, skip_pointer = skip_pointer, save_path = output_path, columns = gps_cols)
        skip_pointer = len(gps_dfs[0])
        
         # overwrite progress text file
        counter += 1
        with open(f"{output_path}/progress.txt", "w") as f:
            f.write(str(counter))
    
    # sync messages # *: only do this once all the msgs synceds are read
    print(f'image_dfs: {len(image_dfs)}, gps_dfs: {len(gps_dfs)}')
    msgs = image_dfs + gps_dfs
    msgs_synced: list[np.array] = sync_msgs(msgs)
    msgs_synced_conc = np.concatenate(msgs_synced, axis=1)
    gps_cols_list = [gps_cols[cols] for cols in gps_cols]
    gps_cols_list = [item for sublist in gps_cols_list for item in sublist]
    msgs_df: pd.DataFrame = pd.DataFrame(msgs_synced_conc, columns=images_cols + gps_cols_list)
    
    # postprocessing
    msgs_df = postprocessing(msgs_df, images_cols)
    
    # output synced messages
    save_path = output_path / 'Metadata'
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
    msgs_df.to_csv(f"{save_path}/msgs_synced.csv", index=False)
    
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--file_names', type=str, nargs='+', required=True,
                    help='List of paths to the event files.')
    ap.add_argument('--output_path', type=str, required=True,
                    help='Path to the folder where the converted data will be written.')

    args = ap.parse_args()
    file_names = [Path(f) for f in args.file_names]
    output_path = Path(args.output_path)

    # Check that all file paths exist
    for f in file_names:
        if not f.exists():
            raise RuntimeError(f"File {f} does not exist.")
    
    # Make output directory
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    # Extract binary files
    extract_binary(file_names, output_path)