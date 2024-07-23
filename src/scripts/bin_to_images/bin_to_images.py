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
from datetime import datetime
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
IMAGE_TYPES = ['rgb','disparity']
GPS_TYPES = ['pvt','relposned']
CALIBRATION = ['calibration']
TYPES = IMAGE_TYPES + GPS_TYPES + CALIBRATION

# gps data to analyze
GPS_PVT = ['stamp','gps_time','longitude','latitude','altitude','heading_motion',
            'heading_accuracy','speed_accuracy','horizontal_accuracy','vertical_accuracy',
            'p_dop','height']
GPS_REL = ['stamp','gps_time','relative_pose_north','relative_pose_east','relative_pose_down',
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
    
    # create gps interpolation function for each gps df
    for key, gps in gps_dfs.items():
        gps_interp_fn = interp1d(
            x = gps[skip_pointer:, 0],
            y = gps[skip_pointer:, 1:],
            axis = 0,
            kind='linear',
            fill_value='extrapolate'
        )
        
        # interpolate gps data at first column of image array
        interp_cols = np.array([gps_interp_fn(value) for value in image_dfs[0][skip_pointer:, 0]])
        
        # add interpolated gps data to image array
        gps_df = np.concatenate([image_dfs[0][skip_pointer:, 0].reshape(-1, 1), interp_cols], axis=1)
        gps_dfs_list.append(gps_df)
        
        # update gps csv file
        gps_dfs_new = pd.DataFrame(gps_df, columns=columns[key])
        if key == 'pvt':
            # check if existing gps data exists
            gps_df_old = pd.read_csv(f"{save_path}/gps_{key}.csv") if (save_path / f"gps_{key}.csv").exists() else pd.DataFrame(columns=GPS_PVT)
        elif key == 'relposned':
            gps_df_old = pd.read_csv(f"{save_path}/gps_{key}.csv") if (save_path / f"gps_{key}.csv").exists() else pd.DataFrame(columns=GPS_REL)
        gps_df_combined = pd.concat([gps_df_old, gps_dfs_new], ignore_index=True)
        gps_df_combined.to_csv(f"{save_path}/gps_{key}.csv", index=False)
            
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
    msgs_df['direction'] = msgs_df['heading_motion'].apply(heading_to_direction)
    
    # rename lat/lon columns
    msgs_df.rename(columns={'longitude': 'lon', 'latitude': 'lat'}, inplace=True)
    
    # filter dataframe
    cols_to_keep = images_cols_new + ['direction'] + ['lat'] + ['lon']
    msgs_df = msgs_df[cols_to_keep]
    
    return msgs_df

# Zhenghao Fei, PAIBL 2020
def sync_msgs(
    msgs: List[np.array], 
    dt_threshold=None
) -> List[np.array]:
    """Written by Zhenghao Fei, PAIBL 2020
    Syncs multiple messages based on their time stamps
    `msgs` should be a numpy array of size (N, data), timestamps should be the first dimension of the msgs
    Synchronization will be based on the first msg in the list

    Args:
        msgs (list[np.array]): Messages to sync with timestamp in the first columns
        dt_threshold (_type_, optional): Defaults to None.

    Returns:
        list[np.array]: final messages synced
    """    
    if dt_threshold is None:
        # if dt is not set, dt will be the average period of the first msg
        msg_t = msgs[0][:, 0]
        dt_threshold = (msg_t[-1] - msg_t[1])/ len(msg_t)
    msg1_t = msgs[0][:, 0]

    # timestamp kd of the rest msgs
    timestamps_kd_list = []
    for msg in msgs[1:]:
        timestamps_kd = KDTree(np.asarray(msg[:, 0]).reshape(-1, 1))
        timestamps_kd_list.append(timestamps_kd)

    msgs_idx_synced = []
    for msg1_idx in range(len(msg1_t)):
        msg_idx_list = [msg1_idx]
        dt_valid = True
        for timestamps_kd in timestamps_kd_list:
            dt, msg_idx = timestamps_kd.query([msg1_t[msg1_idx]])
            if abs(dt) > dt_threshold:
                dt_valid = False
                break
            msg_idx_list.append(msg_idx)

        if dt_valid:
            msgs_idx_synced.append(msg_idx_list)

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
    current_ts: int
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
                
                # check if calibrations exist for this camera
                if not camera_name in calibrations: # ! this should not happen (place holder for now)
                    continue
                
                points_xyz = process_disparity(img, calibrations[camera_name])
                img_name: str = f"disparity-{updated_ts}.npy"
                np.save(str(camera_path / img_name), points_xyz)
            else:
                img_name: str = f"rgb-{updated_ts}.jpg"
                cv2.imwrite(str(camera_path / img_name), img)

    # split dataframe based on columns
    dfs = []
    ts_cols_list = []
    unique_camera_ids = {s.split('/')[1] for s in image_topics if s.startswith('/oak')}
    for i in unique_camera_ids:
        i = CAMERA_POSITIONS[i]
        ts_cols = [f'/{i}/rgb',f'/{i}/disparity']
                                # f'/{i}/left', f'/{i}/right']
        ts_df_split = ts_df[ts_cols]
        ts_df_split = ts_df_split.dropna(subset=[f'/{i}/rgb', f'/{i}/disparity'])
        
        # check if existing ts_df_split exists and concatenate
        if (save_path / f"{i}_timestamps.csv").exists():
            ts_df_existing = pd.read_csv(f"{save_path}/{i}_timestamps.csv")
            ts_df_split = pd.concat([ts_df_existing, ts_df_split], ignore_index=True)

        # output dataframe as csv
        ts_df_split.to_csv(f"{save_path}/{i}_timestamps.csv", index=False) # *concantenate existing ts_df to this one
        dfs.append(ts_df_split.to_numpy(dtype='float64'))
        ts_cols_list += ts_cols
        
    return dfs, ts_cols_list

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
    """ 

    print('--- gps extraction ---')
    
    df = {}
    gps_cols_list = {}
    
    # initialize save path
    save_path = output_path / 'Metadata'
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    # loop through each topic
    for topic_name in gps_topics:

        # create dataframe for this topic # *retrieve existing gps dataframe if it exists
        gps_name = topic_name.split('/')[2]
        if gps_name == 'pvt':
            # check if existing gps data exists
            gps_df = pd.read_csv(f"{save_path}/gps_{gps_name}.csv") if (save_path / f"gps_{gps_name}.csv").exists() else pd.DataFrame(columns=GPS_PVT)
        elif gps_name == 'relposned':
            gps_df = pd.read_csv(f"{save_path}/gps_{gps_name}.csv") if (save_path / f"gps_{gps_name}.csv").exists() else pd.DataFrame(columns=GPS_REL)
        else:
            print('Unknown topic name.')
            return False

        # initialize gps events and event log
        gps_events: list[EventLogPosition] = events_dict[topic_name]
        event_log: EventLogPosition

        # check event log to extract information
        for event_log in tqdm(gps_events):
            
            # read message
            if gps_name == 'pvt':
                sample: gps_pb2.GpsFrame = event_log.read_message()
            elif gps_name == 'relposned':
                sample: gps_pb2.RelativePositionFrame = event_log.read_message()
            else:
                print('Unknown protocol message.')
                return False
            
            # add information into dataframe
            updated_ts = int(current_ts + (sample.stamp.stamp*1e6)) # update timestamp
            if gps_name == 'pvt':
                new_row = {
                    'stamp': [updated_ts], 'gps_time': [sample.gps_time.stamp],
                    'longitude': [sample.longitude], 'latitude': [sample.latitude],
                    'altitude': [sample.altitude], 'heading_motion': [sample.heading_motion], 
                    'heading_accuracy': [sample.heading_accuracy], 'speed_accuracy': [sample.speed_accuracy], 
                    'horizontal_accuracy': [sample.horizontal_accuracy], 'vetical_accuracy': [sample.vertical_accuracy], 
                    'p_dop': [sample.p_dop], 'height': [sample.height]
                }
            elif gps_name == 'relposned':
                new_row = {
                    'stamp': [updated_ts], 'gps_time': [sample.gps_time.stamp],
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
            gps_df = pd.concat([gps_df, new_df], ignore_index=True) # *concatenate new dataframes to existing dataframe

        # output dataframe as csv (or rewrite existing one)
        gps_df.replace({'True': 1, 'False': 0}, inplace=True)
        gps_df = gps_df.apply(pd.to_numeric, errors='coerce')
        gps_df.to_csv(f"{save_path}/gps_{gps_name}.csv", index=False)
        # df.append(gps_df.to_numpy(dtype='float64'))
        df[gps_name] = gps_df.to_numpy(dtype='float64')
        gps_cols_list[gps_name] = gps_df.columns.tolist()

    return df, gps_cols_list

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

def extract_binary(
    file_names: Path,
    output_path: Path
) -> None:
    """Read an events file and extracts relevant information from it.

    Args:

        file_name (Path): The path to the events file.
        output_path (Path): The path to the folder where the converted data will be written.
        disparity_scale (int): Scale for amplifying disparity color mapping. Default: 1.
    """
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
    
    # extract each bin file
    for file_name in tqdm(file_names):
        
        # overwrite progress text file
        counter += 1
        with open(f"{output_path}/progress.txt", "w") as f:
            f.write(str(counter))
        
        # create the file reader
        reader = EventsFileReader(file_name)
        success: bool = reader.open()
        if not success:
            raise RuntimeError(f"Failed to open events file: {file_name}")

        # get the index of the events file
        events_index: list[EventLogPosition] = reader.get_index()

        # structure the index as a dictionary of lists of events
        events_dict: dict[str, list[EventLogPosition]] = build_events_dict(events_index)
        all_topics = list(events_dict.keys())
        print(f"All available topics: {sorted(events_dict.keys())}")

        # keep only relevant topics
        topics = [topic for topic in all_topics if any(type_.lower() in topic.lower() for type_ in TYPES)]
    
        # *add loop going through each .bin file and update progress.txt
        # get datetime of recording 
        if len(os.path.basename(file_name).split('_')) < 7:
            raise RuntimeError(f"'File name is not compatible with this script.")
        date_contents = os.path.basename(file_name).split('_')[:-1]
        date_string = '_'.join(date_contents)
        date_format = '%Y_%m_%d_%H_%M_%S_%f'
        date_object = datetime.strptime(date_string, date_format)
        current_ts = int(date_object.timestamp() * 1e6) # in microseconds # *: this changes for each .bin file
        
        # extract calibration topics
        calib_topics = [topic for topic in topics if any(type_.lower() in topic.lower() for type_ in CALIBRATION)]
        calibrations: dict[str, dict] = extract_calibrations(calib_topics, events_dict, output_path)
        if len(calibrations) == 0:
            raise RuntimeError(f"Failed to extract calibration event file")

        # extract gps topics
        gps_topics = [topic for topic in topics if any(type_.lower() in topic.lower() for type_ in GPS_TYPES)]
        gps_dfs, gps_cols = extract_gps(gps_topics, events_dict, output_path, current_ts)
        if len(gps_dfs) == 0:
            raise RuntimeError(f"Failed to extract gps event file")

        # extract image topics
        image_topics = [topic for topic in topics if any(type_.lower() in topic.lower() for type_ in IMAGE_TYPES)]
        image_dfs, images_cols = extract_images(image_topics, events_dict, calibrations, output_path, current_ts)
        if len(image_dfs) == 0:
            raise RuntimeError(f"Failed to extract image event file")
        
        # *: Interpolate GPS data to query at camera timestamps
        gps_dfs = interpolate_gps(gps_dfs = gps_dfs, image_dfs = image_dfs, skip_pointer = skip_pointer, save_path = output_path, columns = gps_cols)
        skip_pointer = len(gps_dfs[0])
    
    # sync messages # *: only do this once all the msgs synceds are read
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
    
    
# if __name__ == '__main__':

#     ap = argparse.ArgumentParser()
#     ap.add_argument("--file-names", type=Path, nargs='+', required=True,
#         help="Path(s) to the events.bin file exported using the recorder app")
#     ap.add_argument("--output-path", type=Path, required=True,
#         help="Path to output extracted files")
#     args = ap.parse_args()

#     base = 'RGB'
#     main(args.file_names, args.output_path / base)