# Standard library imports
import os
import re
import subprocess
import threading
import time
import glob
import yaml
import random
import string
import csv
import shutil
import traceback
import argparse
import select
import multiprocessing
import requests
from multiprocessing import active_children, Process
from pathlib import Path

# Third-party library imports
import uvicorn
import json
import pandas as pd
import geopandas as gpd
import numpy as np
from flask import Flask, make_response, send_from_directory, jsonify, request, send_file
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.middleware.cors import CORSMiddleware
from werkzeug.utils import secure_filename
import rasterio
from PIL import Image
import io

# Local application/library specific imports
from scripts.drone_trait_extraction import shared_states
from scripts.drone_trait_extraction.drone_gis import process_tiff, find_drone_tiffs, query_drone_images
from scripts.orthomosaic_generation import run_odm, reset_odm, make_odm_args, convert_tif_to_png
from scripts.utils import process_directories_in_parallel
from scripts.gcp_picker import collect_gcp_candidate, get_image_exif, refresh_gcp_candidate
from scripts.bin_to_images.bin_to_images import extract_binary

# Paths to scripts
TRAIN_MODEL = os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts/deep_learning/model_training/train.py'))
LOCATE_PLANTS = os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts/deep_learning/trait_extraction/locate.py'))
EXTRACT_TRAITS = os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts/deep_learning/trait_extraction/extract.py'))

# Define the Flask application for serving files
file_app = Flask(__name__)
latest_data = {'epoch': 0, 'map': 0, 'locate': 0, 'extract': 0, 'ortho': 0, 'drone_extract': 0}
training_stopped_event = threading.Event()
extraction_processes = {}
extraction_status = "not_started"  # Possible values: not_started, in_progress, done, failed

def process_exif_data_async(file_paths, data_type, msgs_synced_file, existing_df, existing_paths):
    exif_data_list = []
    
    # Extract EXIF Data Extraction
    for file_path in file_paths:
        if data_type.lower() == 'image':
            if file_path not in existing_paths:
                msg = get_image_exif(file_path)
                if msg and msg['image_path'] not in existing_paths:
                    exif_data_list.append(msg)
                    existing_paths.add(msg['image_path'])  # Prevent duplicated process
    
    if data_type.lower() == 'image' and exif_data_list:
        if existing_df is not None and not existing_df.empty:
            pd.DataFrame(exif_data_list).to_csv(msgs_synced_file, mode='a', header=False, index=False)
        else:
            pd.DataFrame(exif_data_list).to_csv(msgs_synced_file, mode='w', header=True, index=False)

@file_app.route('/get_tif_to_png', methods=['POST'])
def get_tif_to_png():
    data = request.json
    tif_path = data['filePath']
    
    # Construct the full file paths
    tif_full_path = os.path.join(data_root_dir, tif_path)
    
    # Generate PNG path by replacing .tif extension
    png_path = tif_full_path.replace('.tif', '.png')
    
    if not os.path.exists(png_path):
        try:
            print(f"Converting {tif_full_path} to {png_path}")
            # Convert the TIF file to PNG
            convert_tif_to_png(tif_full_path)
        except Exception as e:
            print(f"Error converting TIF to PNG: {e}")
            return jsonify({'error': str(e)}), 500
    
    try:
        # Open and send the existing PNG file
        return send_file(png_path, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

#### FILE SERVING ENDPOINTS ####
# endpoint to serve files
@file_app.route('/files/<path:filename>')
def serve_files(filename):
    global data_root_dir
    return send_from_directory(data_root_dir, filename)

# endpoint to serve image in memory
@file_app.route('/images/<path:filename>')
def serve_image(filename):
    global image_dict
    return image_dict[filename]
    
@file_app.route('/fetch_data_root_dir')
def fetch_data_root_dir():
    global data_root_dir
    return data_root_dir

# endpoint to list directories
@file_app.route('/list_dirs/<path:dir_path>', methods=['GET'])
def list_dirs(dir_path):
    global data_root_dir
    dir_path = os.path.join(data_root_dir, dir_path)  # join with base directory path
    if os.path.exists(dir_path):
        dirs = (entry.name for entry in os.scandir(dir_path) if entry.is_dir())
        return jsonify(list(dirs)), 200
    else:
        return jsonify({'message': 'Directory not found'}), 404
    
def stream_output(process):
    """Function to read the process output and errors in real-time."""
    while True:
        reads = [process.stdout.fileno(), process.stderr.fileno()]
        ret = select.select(reads, [], [])

        for fd in ret[0]:
            if fd == process.stdout.fileno():
                output = process.stdout.readline()
                if output:
                    print("Output:", output.decode('utf-8').strip())
            if fd == process.stderr.fileno():
                error_output = process.stderr.readline()
                if error_output:
                    print("Error:", error_output.decode('utf-8').strip())

        if process.poll() is not None:
            break  # Break loop if process ends

    # Close stdout and stderr after reading
    process.stdout.close()
    process.stderr.close()

@file_app.get("/list_dirs_nested")
async def list_dirs_nested():
    base_dir = Path(data_root_dir) / 'Raw'
    combined_structure = await process_directories_in_parallel(base_dir, max_depth=9)
    return jsonify(combined_structure), 200

@file_app.get("/list_dirs_nested_processed")
async def list_dirs_nested_processed():
    base_dir = Path(data_root_dir) / 'Processed'
    combined_structure = await process_directories_in_parallel(base_dir, max_depth=9)
    return jsonify(combined_structure), 200

# endpoint to list files
@file_app.route('/list_files/<path:dir_path>', methods=['GET'])
def list_files(dir_path):
    global data_root_dir
    dir_path = os.path.join(data_root_dir, dir_path)
    if os.path.exists(dir_path):
        files = os.listdir(dir_path)
        files = [x for x in files if not x.startswith('.')]
        files = [x for x in files if not os.path.isdir(os.path.join(dir_path, x))]
        files.sort()
        return jsonify(files), 200
    else:
        return jsonify({'message': 'Directory not found'}), 404

@file_app.route('/view_synced_data', methods=['POST'])
def view_synced_data():
    data = request.get_json()
    base_dir = data.get('base_dir')  # Relative path like: IITA_Test/Nigeria/AmigaSample/2025-04-29/rover/RGB

    if not base_dir:
        return jsonify({'error': 'Missing base_dir'}), 400

    # Construct full path to msgs_synced.csv
    full_path = os.path.join(data_root_dir, base_dir, 'Metadata', 'msgs_synced.csv')
    print(f"Full path to msgs_synced.csv: {full_path}")

    if not os.path.exists(full_path):
        return jsonify({'error': 'File not found'}), 404

    try:
        with open(full_path, newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        return jsonify({'data': rows})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@file_app.route('/restore_images', methods=['POST'])
def restore_images():
    global data_root_dir

    data = request.get_json()
    image_names = data.get('images')
    removed_dir = data.get('removed_dir')  # e.g. Raw/.../Removed/top

    if not image_names or not removed_dir:
        return jsonify({'error': 'Missing images or removed_dir'}), 400

    # Full path to source in Removed/
    removed_path = os.path.join(data_root_dir, removed_dir)

    # Replace 'Removed' with 'Images' to get destination
    if 'Removed' not in removed_dir:
        return jsonify({'error': "'Removed' folder not found in path"}), 400

    restored_dir_rel = removed_dir.replace('/Removed/', '/Images/')
    restored_dir_abs = os.path.join(data_root_dir, restored_dir_rel)
    os.makedirs(restored_dir_abs, exist_ok=True)

    moved = []
    errors = []

    for image_name in image_names:
        src = os.path.join(removed_path, image_name)
        dst = os.path.join(restored_dir_abs, image_name)

        try:
            if os.path.exists(src):
                shutil.move(src, dst)
                moved.append(image_name)
            else:
                errors.append(f"{image_name} not found in Removed.")
        except Exception as e:
            errors.append(f"{image_name} failed: {str(e)}")

    return jsonify({
        'restored': moved,
        'restored_dir': restored_dir_rel,
        'errors': errors
    }), 200

@file_app.route('/remove_images', methods=['POST'])
def remove_images():
    global data_root_dir

    data = request.get_json()
    image_names = data.get('images')
    source_dir = data.get('source_dir')  # e.g. Raw/.../Images/top

    if not image_names or not source_dir:
        return jsonify({'error': 'Missing images or source_dir'}), 400

    # Full path to source
    source_path = os.path.join(data_root_dir, source_dir)

    # Replace 'Images' with 'Removed' in the path
    if 'Images' not in source_dir:
        return jsonify({'error': "'Images' folder not found in path"}), 400

    removed_dir_rel = source_dir.replace('/Images/', '/Removed/')
    removed_dir_abs = os.path.join(data_root_dir, removed_dir_rel)
    os.makedirs(removed_dir_abs, exist_ok=True)

    moved = []
    errors = []

    for image_name in image_names:
        src = os.path.join(source_path, image_name)
        dst = os.path.join(removed_dir_abs, image_name)

        try:
            if os.path.exists(src):
                shutil.move(src, dst)
                moved.append(image_name)
            else:
                errors.append(f"{image_name} not found.")
        except Exception as e:
            errors.append(f"{image_name} failed: {str(e)}")

    return jsonify({
        'moved': moved,
        'removed_dir': removed_dir_rel,
        'errors': errors
    }), 200

@file_app.route('/update_data', methods=['POST'])
def update_data():
    try:
        data = request.json
        old_data = data['oldData']
        new_data = data['updatedData']
        prefix = data_root_dir

        new_path_raw = os.path.join(prefix, 'Raw', new_data['year'], new_data['experiment'], new_data['location'], new_data['population'], new_data['date'], new_data['platform'], new_data['sensor'])
        new_path_intermediate = os.path.join(prefix, 'Intermediate', new_data['year'], new_data['experiment'], new_data['location'], new_data['population'], new_data['date'], new_data['platform'], new_data['sensor'])
        new_path_processed = os.path.join(prefix, 'Processed', new_data['year'], new_data['experiment'], new_data['location'], new_data['population'], new_data['date'], new_data['platform'], new_data['sensor'])
        
        old_path_raw = os.path.join(prefix, 'Raw', old_data['year'], old_data['experiment'], old_data['location'], old_data['population'], old_data['date'], old_data['platform'], old_data['sensor'])
        old_path_intermediate = os.path.join(prefix, 'Intermediate', old_data['year'], old_data['experiment'], old_data['location'], old_data['population'], old_data['date'], old_data['platform'], old_data['sensor'])
        old_path_processed = os.path.join(prefix, 'Processed', old_data['year'], old_data['experiment'], old_data['location'], old_data['population'], old_data['date'], old_data['platform'], old_data['sensor'])

        # Rename paths
        print("Renaming directories...")
        if os.path.exists(old_path_raw):
                os.rename(old_path_raw, new_path_raw)
        if os.path.exists(old_path_intermediate):
                os.rename(old_path_intermediate, new_path_intermediate)
        if os.path.exists(old_path_processed):
                os.rename(old_path_processed, new_path_processed)
            
        # print('Making new directory...')
        # os.makedirs(new_path_raw, exist_ok=True)
        # os.makedirs(new_path_intermediate, exist_ok=True)
        # os.makedirs(new_path_processed, exist_ok=True)

        # # Move files from old directory to new directory
        # print('Moving files...')
        # for folder in ['Raw', 'Intermediate', 'Processed']:
        #     old_dir = os.path.join(prefix, folder, old_data['year'], old_data['experiment'], old_data['location'], old_data['population'], old_data['date'], old_data['platform'], old_data['sensor'])
        #     new_dir = os.path.join(prefix, folder, new_data['year'], new_data['experiment'], new_data['location'], new_data['population'], new_data['date'], new_data['platform'], new_data['sensor'])

        #     if os.path.exists(old_dir):
        #         for item in os.listdir(old_dir):
        #             old_item_path = os.path.join(old_dir, item)
        #             new_item_path = os.path.join(new_dir, item)
        #             shutil.move(old_item_path, new_item_path)
                
        #         def is_empty_dir(path):
        #             return all(os.path.isdir(os.path.join(path, d)) and len(os.listdir(os.path.join(path, d))) == 0
        #                        for d in os.listdir(path))
                
        #         while os.path.exists(old_dir) and is_empty_dir(old_dir):
        #             try:
        #                 os.rmdir(old_dir)
        #                 old_dir = os.path.dirname(old_dir)
        #             except OSError:
        #                 break
        npy_path = new_path_raw + "/image_names_final.npy"

        if not os.path.isfile(npy_path):
            print(f"The file {npy_path} does not exist. Skipping this block.")
        else:
            loaded_npy = np.load(npy_path, allow_pickle=True).item()
            new_path_npy = f"/Raw/{new_data['year']}/{new_data['experiment']}/{new_data['location']}/{new_data['population']}/{new_data['date']}/{new_data['platform']}/{new_data['sensor']}/Images/"
            for entry in loaded_npy['selected_images']:
                entry['image_path'] = new_path_npy + entry['image_path'].split('/')[-1]
            np.save(npy_path, loaded_npy)

        for file in os.listdir(new_path_processed):
            file_path = os.path.join(new_path_processed, file)
            if os.path.isfile(file_path) and file.lower().endswith('.tif'):
                base_name, ext = os.path.splitext(file)
                last_part = "-".join(base_name.split('-')[3:])
                new_filename = f"{new_data['date']}-{last_part}{ext}"
                new_file_path = os.path.join(new_path_processed, new_filename)
                os.rename(file_path, new_file_path)
        

        return jsonify({'message': 'Directories updated successfully.'}), 200

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

@file_app.route('/delete_files', methods=['POST'])
def delete_files():
    try:
        data = request.json
        data_to_del = data['data_to_del']
        prefix = data_root_dir

        paths = {
            'Raw': os.path.join(prefix, 'Raw', data_to_del['year'], data_to_del['experiment'], data_to_del['location'], data_to_del['population'], data_to_del['date'], data_to_del['platform'], data_to_del['sensor']),
            'Intermediate': os.path.join(prefix, 'Intermediate', data_to_del['year'], data_to_del['experiment'], data_to_del['location'], data_to_del['population'], data_to_del['date'], data_to_del['platform'], data_to_del['sensor']),
            'Processed': os.path.join(prefix, 'Processed', data_to_del['year'], data_to_del['experiment'], data_to_del['location'], data_to_del['population'], data_to_del['date'], data_to_del['platform'], data_to_del['sensor'])
        }

        for _, path in paths.items():
            if os.path.exists(path):
                for item in os.listdir(path):
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                
                try:
                    os.rmdir(path)
                except OSError:
                    pass

                current_dir = os.path.dirname(path)
                while os.path.exists(current_dir) and not os.listdir(current_dir):
                    try:
                        os.rmdir(current_dir)
                        current_dir = os.path.dirname(current_dir)
                    except OSError:
                        break

        return jsonify({'message': 'Files and directories deleted successfully.'}), 200

    except Exception as e:
        return jsonify({'message': 'Cache directory not found'}), 404
    
@file_app.route('/best_locate_file', methods=['POST'])
def get_best_locate_file():
    try:
        data = request.json
        if not data or len(data) == 0:
            return jsonify(None), 200  # No locate files provided, return None
        
        print(f"All locate files: {data}")
        
        # Pattern to match Locate ID
        pattern = r'Locate-([^/]+)/locate\.csv'
        
        if len(data) == 1:
            # Handle case when there's only one locate file
            best_locate = data[0]
            match = re.search(pattern, best_locate)
            if match:
                best_locate_match = match.group(1)
                return jsonify(best_locate_match), 200  # Return the ID of the single locate file
            else:
                return jsonify(None), 200  # No match found, return None
        else:
            # Handle case when there are multiple locate files
            best_locate_match = None
            locate_matches = {}
            for locate_file in data:
                match = re.search(pattern, locate_file)
                if match:
                    locate_id = match.group(1)
                    base_path = Path(os.path.dirname(locate_file))
                    date = base_path.parts[-5]
                    platform = base_path.parts[-4]
                    sensor = base_path.parts[-3]
                    
                    # get mAP of model
                    date_index = base_path.parts.index(date[0]) if date[0] in base_path.parts else None
                    if date_index and date_index > 0:
                        # Construct a new path from the parts up to the folder before the known date
                        root_path = Path(*base_path.parts[:date_index])
                        results_file = root_path / 'Training' / platform / f'{sensor} Plant Detection' / f'Plant-{model_id[0]}' / 'results.csv'
                    df = pd.read_csv(results_file, delimiter=',\s+', engine='python')
                    mAP = round(df['metrics/mAP50(B)'].iloc[-1], 2)
                    locate_matches[locate_id] = mAP
                    
            # get the locate file with the highest mAP
            if locate_matches:
                best_locate_match = max(locate_matches, key=locate_matches.get)
            
            return jsonify(best_locate_match), 200  # Return the first matching locate ID or None if no match

    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Catch-all for unexpected errors

@file_app.route('/best_model_file', methods=['POST'])
def get_best_model_file():
    
    try:
        
        data = request.json
        if not data or len(data) == 0:
            return jsonify(None), 200
        
        if len(data) == 1:
            return jsonify(data[0]), 200
        
        else:
            best_model = None
            model_matches = {}
            for model_file in data:
                base_path = Path(model_file).parent.parent
                run = base_path.name
                match = re.search(r'-([A-Za-z0-9]+)$', run)
                id = match.group(1)
                results_file = base_path / 'results.csv'

                # get mAP of model
                df = pd.read_csv(results_file, delimiter=',\s+', engine='python')
                mAP = round(df['metrics/mAP50(B)'].iloc[-1], 2)  # Get the last value in the column
                model_matches[id] = mAP
                
            # get the model file with the highest mAP
            if model_matches:
                best_model = max(model_matches, key=model_matches.get)
                
            return jsonify(best_model), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@file_app.route('/check_runs/<path:dir_path>', methods=['GET'])
def check_runs(dir_path):
    global data_root_dir
    dir_path = os.path.join(data_root_dir, dir_path)
    response_data = {}  # Initialize an empty dictionary for the response
    
    # For the Model column of Locate Plants
    if os.path.exists(dir_path) and 'Plant Detection' in dir_path:
        check = f'{dir_path}/Plant-*/weights/last.pt'
        matched_paths = glob.glob(check)
        
        for path in matched_paths:
            # Construct the path to the logs.yaml file in the same directory as best.pt
            logs_yaml_path = os.path.join(os.path.dirname(os.path.dirname(path)), 'logs.yaml')
            
            # Initialize an empty list for dates; it will remain empty if logs.yaml is not found or cannot be parsed
            dates = []
            
            # Check if logs.yaml exists
            if os.path.exists(logs_yaml_path):
                try:
                    # Open and parse the logs.yaml file
                    with open(logs_yaml_path, 'r') as file:
                        logs_data = yaml.safe_load(file)
                        # Extract dates if available
                        dates = logs_data.get('dates', [])
                except yaml.YAMLError as exc:
                    print(f"Error parsing YAML file {logs_yaml_path}: {exc}")
            
            # Update the response_data dictionary with the path and its corresponding dates
            response_data[path] = dates
            
    # For the Locate column of Locate Plants
    elif os.path.exists(dir_path) and 'Locate' in dir_path:
        check = f'{dir_path}/Locate-*/locate.csv'
        response_data = glob.glob(check)
        
    # For Labels Sets column of Teach Traits
    elif os.path.exists(dir_path) and 'Labels' in dir_path:
        date_pattern = r"\d{4}-\d{2}-\d{2}"
        match = re.search(date_pattern, dir_path)
        extracted_date = match.group(0) if match else None
        response_data = {extracted_date: dir_path}
        
    # For Models column of Teach Traits
    elif os.path.exists(dir_path) and any(x in dir_path for x in ['Pod', 'Flower', 'Leaf']):
        check = f'{dir_path}/**/weights/best.pt'
        matched_paths = glob.glob(check, recursive=True)
        
        for path in matched_paths:
            # Construct the path to the logs.yaml file in the same directory as best.pt
            logs_yaml_path = os.path.join(os.path.dirname(os.path.dirname(path)), 'logs.yaml')
            
            # Initialize an empty list for dates; it will remain empty if logs.yaml is not found or cannot be parsed
            dates = []
            
            # Check if logs.yaml exists
            if os.path.exists(logs_yaml_path):
                try:
                    # Open and parse the logs.yaml file
                    with open(logs_yaml_path, 'r') as file:
                        logs_data = yaml.safe_load(file)
                        # Extract dates if available
                        dates = logs_data.get('dates', [])
                except yaml.YAMLError as exc:
                    print(f"Error parsing YAML file {logs_yaml_path}: {exc}")
            
            details = check_model_details(Path(path))
            details['dates'] = dates
            
            # Update the response_data dictionary with the path and its corresponding dates
            response_data[path] = details
            
    elif os.path.exists(dir_path) and 'Processed' in dir_path:
        logs = f'{dir_path}/logs.yaml'
        # if log files exist, read the log files else return empty dictionary
        if os.path.exists(logs):
            with open(logs, 'r') as file:
                data = yaml.safe_load(file)
                response_data = {k: {'model': v['model'], 'locate': v['locate'], 'id': v['id']} for k, v in data.items()}
        
    return jsonify(response_data), 200
    
@file_app.route('/upload', methods=['POST'])
def upload_files():
    data_type = request.form.get('dataType')
    dir_path = request.form.get('dirPath')
    upload_new_files_only = request.form.get('uploadNewFilesOnly') == 'true'
    full_dir_path = os.path.join(UPLOAD_BASE_DIR, dir_path)
    os.makedirs(full_dir_path, exist_ok=True)
    
    # Read msgs_synced.csv once if it's an image upload
    existing_paths = set()
    msgs_synced_file = None
    existing_df = None
    
    if data_type == 'image':
        msgs_synced_file = os.path.join(os.path.dirname(full_dir_path), "msgs_synced.csv")
        if os.path.isfile(msgs_synced_file):
            existing_df = pd.read_csv(msgs_synced_file)
            existing_paths = set(existing_df['image_path'].values)

    uploaded_file_paths = [] 

    for file in request.files.getlist("files"):
        filename = secure_filename(file.filename)
        if data_type == 'fieldDesign':
            filename = 'field_design.csv'
        elif data_type == 'gcpLocations':
            filename = 'gcp_locations.csv'
        
        file_path = os.path.join(full_dir_path, filename)

        if upload_new_files_only and os.path.isfile(file_path):
            print(f"Skipping {filename} because it already exists in {dir_path}")
        else:
            file.save(file_path)
            uploaded_file_paths.append(file_path)  

    
    if data_type.lower() == 'image' and uploaded_file_paths:
        thread = threading.Thread(
            target=process_exif_data_async, 
            args=(uploaded_file_paths, data_type, msgs_synced_file, existing_df, existing_paths)
        )
        thread.daemon = True  # Stop the main thread
        thread.start()

    return jsonify({'message': 'Files uploaded successfully'}), 200

@file_app.route('/check_files', methods=['POST'])
def check_files():
    data = request.json
    fileList = data['fileList']
    dirPath = data['dirPath']
    full_dir_path = os.path.join(UPLOAD_BASE_DIR, dirPath)

    existing_files = set(os.listdir(full_dir_path)) if os.path.exists(full_dir_path) else set()
    new_files = [file for file in fileList if file not in existing_files]

    print(f"Uploading {str(len(new_files))} out of {str(len(fileList))} files to {dirPath}")

    return jsonify(new_files), 200

@file_app.route("/get_binary_report", methods=["POST"])
def get_binary_report():
    data = request.json
    # Construct file path based on metadata
    report_path = f"{UPLOAD_BASE_DIR}/{data['year']}/{data['experiment']}/{data['location']}/{data['population']}/{data['date']}/rover/RGB/report.txt"
    
    try:
        with open(report_path, "r") as f:
            content = f.read()
        return content, 200
    except Exception as e:
        return f"Error loading report: {str(e)}", 500

@file_app.route('/cancel_extraction', methods=['POST'])
def cancel_extraction():
    data = request.json
    dir_path = data.get('dirPath')
    p = extraction_processes.pop(dir_path, None)
    if not p:
        return jsonify({'status': 'no active extraction for this path'}), 404

    p.terminate()  # force-kill the worker
    p.join()
    return jsonify({'status': 'cancelled'}), 200

def _cleanup_files(file_paths):
    global extraction_status
    for p in file_paths:
        try:
            os.remove(p)
        except OSError:
            pass

def _extraction_worker(file_paths, output_path):
    global extraction_status

    try:
        extraction_status = "in_progress"
        extract_binary(file_paths, output_path)
        
        # cleanup files
        _cleanup_files(file_paths)
        extraction_status = "done"
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        # print full traceback
        traceback.print_exc()
        extraction_status = "failed"

@file_app.route('/get_binary_status', methods=['GET'])
def get_binary_status():
    print(f"Extraction status: {extraction_status}")
    return jsonify({'status': extraction_status}), 200

@file_app.route('/extract_binary_file', methods=['POST'])
def extract_binary_file():
    data = request.json
    files = [secure_filename(f) for f in data['files']]
    dir_path = data['localDirPath']
    file_paths = [str(Path(UPLOAD_BASE_DIR) / dir_path / f) for f in files]
    output_path = Path(UPLOAD_BASE_DIR) / dir_path
    
    def extract_timestamp(filename):
        match = re.match(r"(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d+)", filename)
        return match.group(1) if match else filename  # fallback to filename if no match

    # Sort files by timestamp before constructing paths
    files_sorted = sorted(files, key=extract_timestamp)
    file_paths = [str(Path(UPLOAD_BASE_DIR) / dir_path / f) for f in files_sorted]

    print(f"Extracting binary files: {file_paths}")

    # Start new background process
    global extraction_status
    # if extraction_status == "in_progress":
    #     print("Extraction already in progress")
    #     return jsonify({'status': 'already running'}), 429

    extraction_status = "in_progress"
    p = Process(target=_extraction_worker, args=(file_paths, output_path), daemon=True)
    p.start()
    extraction_processes[dir_path] = p

    return jsonify({'status': 'started'}), 200

@file_app.route('/get_binary_progress', methods=['POST'])
def get_binary_progress():
    dir_path = request.json['localDirPath']
    dir_path = os.path.join(UPLOAD_BASE_DIR, dir_path)
    
    try:
        # Traverse through the directory and its subdirectories
        for root, dirs, files in os.walk(dir_path):
            if 'progress.txt' in files:
                progress_file_path = os.path.join(root, 'progress.txt')
                # print(f'progress.txt found in {progress_file_path}')
                with open(progress_file_path, 'r') as file:
                    progress = int(file.read().strip())
                print(f'Binary extraction progress: {progress}')
                return jsonify({'progress': progress}), 200
        
        # If no progress.txt is found
        print(f'progress.txt not found in {progress_file_path}')
        return jsonify({'progress': 0, 'error': 'progress.txt not found'}), 404
    
    except Exception as e:
        return jsonify({'progress': 0, 'error': str(e)}), 500

@file_app.route('/upload_chunk', methods=['POST'])
def upload_chunk():
    chunk = request.files['fileChunk']
    chunk_index = request.form['chunkIndex']
    total_chunks = request.form['totalChunks']
    file_name = secure_filename(request.form['fileIdentifier'])
    dir_path = request.form['dirPath']
    full_dir_path = os.path.join(UPLOAD_BASE_DIR, dir_path)
    cache_dir_path = os.path.join(full_dir_path, 'cache')
    os.makedirs(full_dir_path, exist_ok=True)
    os.makedirs(cache_dir_path, exist_ok=True)
    
    chunk_save_path = os.path.join(cache_dir_path, f"{file_name}.part{chunk_index}")
    chunk.save(chunk_save_path)
    
    # Check if all parts are uploaded
    print(f"Chunk {chunk_index} of {total_chunks} received")
    if all(os.path.exists(os.path.join(cache_dir_path, f"{file_name}.part{i}")) for i in range(int(total_chunks))):
        # Reassemble file
        with open(os.path.join(full_dir_path, file_name), 'wb') as full_file:
            for i in range(int(total_chunks)):
                with open(os.path.join(cache_dir_path, f"{file_name}.part{i}"), 'rb') as part_file:
                    full_file.write(part_file.read())

        time.sleep(60)  # Wait for 60 seconds
        return "File reassembled and saved successfully", 200
    else:
        return f"Chunk {chunk_index} of {total_chunks} received", 202
    
@file_app.route('/check_uploaded_chunks', methods=['POST'])
def check_uploaded_chunks():
    data = request.json
    file_identifier = data['fileIdentifier']
    dir_path = data['localDirPath']
    full_dir_path = os.path.join(UPLOAD_BASE_DIR, dir_path)
    cache_dir_path = os.path.join(full_dir_path, 'cache')
    
    uploaded_chunks = [f for f in os.listdir(cache_dir_path) if f.startswith(file_identifier)]
    uploaded_chunks_count = len(uploaded_chunks)

    return jsonify({'uploadedChunksCount': uploaded_chunks_count}), 200

@file_app.route('/clear_upload_dir', methods=['POST'])
def clear_upload_dir():
    # 1. Grab the user-supplied relative path
    dir_path = request.json.get('dirPath', '').strip()
    if not dir_path:
        return jsonify({'message': 'No directory specified'}), 400
    print(f"Clearing upload directory: {dir_path}")

    # 2. Resolve base and target
    base = Path(UPLOAD_BASE_DIR).resolve()
    target = (base / dir_path).resolve()

    # 3. Safety checks
    #  - must be a subdirectory of base
    #  - must not be equal to base itself
    if not str(target).startswith(str(base) + os.sep):
        return jsonify({'message': 'Invalid path'}), 400
    if target == base:
        return jsonify({'message': 'Refusing to delete root directory'}), 400

    # 4. Delete
    if target.exists():
        try:
            shutil.rmtree(target)
            return jsonify({'message': f'{dir_path} cleared successfully'}), 200
        except Exception as e:
            print(f"Failed to delete {target}: {e}")
            return jsonify({'message': f'Failed to clear dir: {e}'}), 500
    else:
        return jsonify({'message': 'Directory not found'}), 404

@file_app.route('/clear_upload_cache', methods=['POST'])
def clear_upload_cache():
    
    try:
        print('Clearing cache...')
        dir_path = request.json['localDirPath']
        cache_dir_path = os.path.join(UPLOAD_BASE_DIR, dir_path, 'cache')
        
        # loop through each file in cache directory and remove it
        for file in os.listdir(cache_dir_path):
            file_path = os.path.join(cache_dir_path, file)
            os.remove(file_path)
            
        # remove the cache directory
        os.rmdir(cache_dir_path)
        # time.sleep(60)  # Wait for 60 seconds
        return jsonify({'message': 'Cache cleared successfully'}), 200
    except Exception as e:
        print(f'Error clearing cache: {str(e)}')
        return jsonify({'message': 'Cache directory not found'}), 404

#### SCRIPT SERVING ENDPOINTS ####
# endpoint to run script
@file_app.route('/run_script', methods=['POST'])
def run_script():
    data = request.json
    script_path = data.get('script_path')

    def run_in_thread(script_path):
        subprocess.call(script_path, shell=True)

    thread = threading.Thread(target=run_in_thread, args=(script_path,))
    thread.start()

    return jsonify({'message': 'Script started'}), 200


@file_app.route('/get_gcp_selcted_images', methods=['POST'])
def get_gcp_selcted_images():
    global data_root_dir
    try:
        image_folder = os.path.join(data_root_dir, 'Raw', request.json['year'], 
                                    request.json['experiment'], request.json['location'], 
                                    request.json['population'], request.json['date'], 
                                    request.json['platform'], request.json['sensor'], 'Images')
        selected_images = collect_gcp_candidate(data_root_dir, image_folder, request.json['radius_meters'])
        status = "DONE"

        # Return the selected images and their corresponding GPS coordinates
        return jsonify({'selected_images': selected_images,
                        'num_total': len(selected_images),
                        'status':status}), 200
    
    except Exception as e:
        print(e)
        selected_images = []
        status = "DONE"
        return jsonify({'selected_images': selected_images,
                    'num_total': len(selected_images),
                    'status':status}), 200


@file_app.route('/refresh_gcp_selcted_images', methods=['POST'])
def refresh_gcp_selcted_images():
    global data_root_dir
    try:
        image_folder = os.path.join(data_root_dir, 'Raw', request.json['year'], 
                                    request.json['experiment'], request.json['location'], 
                                    request.json['population'], request.json['date'], 
                                    request.json['platform'], request.json['sensor'], 'Images')
        selected_images = refresh_gcp_candidate(data_root_dir, image_folder, request.json['radius_meters'])
        status = "DONE"

        # Return the selected images and their corresponding GPS coordinates
        return jsonify({'selected_images': selected_images,
                        'num_total': len(selected_images),
                        'status':status}), 200
    except Exception as e:
        print(e)
        selected_images = []
        status = "DONE"
        return jsonify({'selected_images': selected_images,
                    'num_total': len(selected_images),
                    'status':status}), 200

@file_app.route('/get_drone_extract_progress', methods=['GET'])
def get_drone_extract_progress():
    global processed_image_folder
    # data = request.json
    # tiff_rgb = data['tiff_rgb']
    # print("Processed image folder: "+ processed_image_folder)
    txt_file = os.path.join(processed_image_folder, 'progress.txt')
    
    # Check if the file exists
    if os.path.exists(txt_file):
        with open(txt_file, 'r') as file:
            number = file.read().strip()
            if number == '':
                latest_data['drone_extract'] = 0
            else:
                latest_data['drone_extract'] = float(number)
        return jsonify(latest_data)
    else:
        return jsonify({'error': 'Drone progress not found'}), 404
    
@file_app.route('/stop_drone_extract', methods=['POST'])
def stop_drone_extract():
    try:
        shared_states.stop_signal = True
        print(f'Shared states variable changed: {shared_states.stop_signal}')
        latest_data['drone_extract'] = 0
        print('Drone Extraction stopped by user.')
        return jsonify({"message": f"Drone Extraction process successfully stopped"}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": e.stderr.decode("utf-8")}), 500
    
@file_app.route('/process_drone_tiff', methods=['POST'])
def process_drone_tiff():
    global now_drone_processing, processed_image_folder
    
    # receive the parameters
    location = request.json['location']
    population = request.json['population']
    date = request.json['date']
    year = request.json['year']
    experimnent = request.json['experiment']
    platform = request.json['platform']
    sensor = request.json['sensor']

    intermediate_prefix = data_root_dir+'/Intermediate'
    processed_prefix = data_root_dir+'/Processed'
    intermediate_image_folder = os.path.join(intermediate_prefix, year, experimnent, location, population)
    processed_image_folder = os.path.join(processed_prefix, year, experimnent, location, population, date, platform, sensor)
    # print("Setting processed image folder: "+ processed_image_folder)
    # Check if already in processing
    if now_drone_processing:
        return jsonify({'message': 'Already in processing'}), 400
    
    now_drone_processing = True
    shared_states.stop_signal = False
    
    try: 
        rgb_tif_file, dem_tif_file, thermal_tif_file = find_drone_tiffs(processed_image_folder)
        geojson_path = os.path.join(intermediate_image_folder,'Plot-Boundary-WGS84.geojson')
        date = processed_image_folder.split("/")[-3]
        output_geojson = os.path.join(processed_image_folder,f"{date}-{platform}-{sensor}-Traits-WGS84.geojson")
        result = process_tiff(tiff_files_rgb=rgb_tif_file,
                     tiff_files_dem=dem_tif_file,
                     tiff_files_thermal=thermal_tif_file,
                     plot_geojson=geojson_path,
                     output_geojson=output_geojson,
                     debug=False)
        now_drone_processing = False
        shared_states.stop_signal = True
        return jsonify({'message': str(result)}), 200

    except Exception as e:
        now_drone_processing = False
        shared_states.stop_signal = True
        print(e)
        return jsonify({'message': str(e)}), 400


@file_app.route('/save_array', methods=['POST'])
def save_array(debug=False):
    data = request.json
    if 'array' not in data:
        return jsonify({"message": "Missing array in data"}), 400

    # Extracting the directory path based on the first element in the array 
    base_image_path = data['array'][0]['image_path']
    platform = data['platform']
    sensor = data['sensor']
    processed_path = os.path.join(base_image_path.replace('/Raw/', 'Intermediate/').split(f'/{platform}')[0], platform, sensor)
    save_directory = os.path.join(data_root_dir, processed_path)
    if debug:
        print(save_directory, flush=True)

    # Creating the directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    filename = os.path.join(save_directory, "gcp_list.txt")

    # Load existing data from file
    existing_data = {}
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.strip().split()
                # Use image name as a key for easy lookup
                image_name = parts[5]
                existing_data[image_name] = {
                    'gcp_lon': parts[0],
                    'gcp_lat': parts[1],
                    'pointX': parts[3],
                    'pointY': parts[4],
                    'image_path': os.path.join(processed_path, image_name),
                    'gcp_label': parts[6],
                    'naturalWidth': parts[7],
                    'naturalHeight': parts[8]
                }

    # Merge new data with existing data
    for item in data['array']:
        if 'pointX' in item and 'pointY' in item:
            if debug:
                print(item, flush=True)
            image_name = item['image_path'].split("/")[-1]
            existing_data[image_name] = {
                'gcp_lon': item['gcp_lon'],
                'gcp_lat': item['gcp_lat'],
                'pointX': item['pointX'],
                'pointY': item['pointY'],
                'image_path': os.path.join(processed_path, image_name),
                'gcp_label': item['gcp_label'],
                'naturalWidth': item['naturalWidth'],
                'naturalHeight': item['naturalHeight']
            }

    # Write merged data to file
    with open(filename, "w") as f:
        f.write('EPSG:4326\n')
        for image_name, item in existing_data.items():
            formatted_data = f"{item['gcp_lon']} {item['gcp_lat']} 0 {item['pointX']} {item['pointY']} {image_name} {item['gcp_label']} {item['naturalWidth']} {item['naturalHeight']} \n"
            f.write(formatted_data)

    return jsonify({"message": f"Array saved successfully in {filename}!"}), 200

@file_app.route('/initialize_file', methods=['POST'])
def initialize_file():
    data = request.json
    if 'basePath' not in data:
        return jsonify({"message": "Missing basePath in data"}), 400

    platform = data['platform']
    sensor = data['sensor']
    processed_path = os.path.join(data['basePath'].replace('/Raw/', 'Intermediate/').split(f'/{sensor}')[0], sensor)
    save_directory = os.path.join(data_root_dir, processed_path)

    # Creating the directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    filename = os.path.join(save_directory, "gcp_list.txt")

    existing_data = []
    if os.path.exists(filename):
        # Read the existing data
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.strip().split()
                existing_data.append({
                    'gcp_lon': parts[0],
                    'gcp_lat': parts[1],
                    'pointX': parts[3],
                    'pointY': parts[4],
                    'image_path': os.path.join(processed_path, parts[5])
                })
    else:
        # Create the file if it doesn't exist
        with open(filename, 'w') as f:
            f.write("EPSG:4326\n")
            pass

    return jsonify({"existing_data": existing_data,
                    "file_path": save_directory}), 200

@file_app.route('/query_traits', methods=['POST'])
def query_traits():
    # receive the parameters
    location = request.json['location']
    population = request.json['population']
    date = request.json['date']
    sensor = request.json['sensor']
    year = request.json['year']
    experiment = request.json['experiment']

    prefix = data_root_dir+'/Processed'
    traitpth = os.path.join(prefix, year, experiment, location, population, date, 'Results', 
                          '-'.join([date, sensor, 'Traits-WGS84.geojson']))

    if not os.path.isfile(traitpth):
        return jsonify({'message': []}), 404
    else:
        gdf = gpd.read_file(traitpth)
        traits = list(gdf.columns)
        extraneous_columns = ['Tier','Bed','Plot','Label','Group','geometry']
        traits = [x for x in traits if x not in extraneous_columns]
        print(traits, flush=True)
        return jsonify(traits), 200
    
def select_middle(df):
    middle_index = len(df) // 2  # Find the middle index
    return df.iloc[[middle_index]]  # Use iloc to select the middle row

def filter_images(geojson_features, year, experiment, location, population, date, platform, sensor, middle_image=False):

    global data_root_dir

    # Construct the CSV path from the state variables
    csv_path = os.path.join(data_root_dir, 'Raw', year, experiment, location, 
                            population, date, 'msgs_synced.csv')
    df = pd.read_csv(csv_path)

    # Convert DataFrame to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon_interp_adj, df.lat_interp_adj))

    # Convert GeoJSON features to GeoDataFrame
    geojson = {'type': 'FeatureCollection', 'features': geojson_features}
    geojson_gdf = gpd.GeoDataFrame.from_features(geojson['features'])

    # Perform spatial join to filter images within the GeoJSON polygons
    filtered_gdf = gpd.sjoin(gdf, geojson_gdf, op='within')

    # If the middle image is specified, only return the middle image per plot
    if middle_image:
        filtered_gdf = filtered_gdf.sort_values(by=sensor+"_time")
        filtered_gdf = filtered_gdf.groupby('Plot').apply(select_middle).reset_index(drop=True)

    # Extract the image names and labels from the filtered GeoDataFrame
    filtered_images = filtered_gdf[sensor+"_file"].tolist()
    filtered_labels = filtered_gdf['Label'].tolist()
    filtered_plots = filtered_gdf['Plot'].tolist()

    # Create a list of dictionaries for the filtered images
    filtered_images_new = []
    for image_name in  filtered_images:
        image_path_abs = os.path.join(data_root_dir, 'Raw', year, experiment, location, population, date, platform, sensor, image_name)
        image_path_rel_to_data_root = os.path.relpath(image_path_abs, data_root_dir)
        filtered_images_new.append(image_path_rel_to_data_root)

    imageDataQuery = [{'imageName': image, 'label': label, 'plot': plot} for image, label, plot in zip(filtered_images_new, filtered_labels, filtered_plots)]

    # Sort the filtered_images by label
    imageDataQuery = sorted(imageDataQuery, key=lambda x: x['label'])

    return imageDataQuery

@file_app.route('/query_images', methods=['POST'])
def query_images():
    data = request.get_json()
    geojson_features = data['geoJSON']
    year = data['selectedYearGCP']
    experiment = data['selectedExperimentGCP']
    location = data['selectedLocationGCP']
    population = data['selectedPopulationGCP']
    date = data['selectedDateQuery']
    sensor = data['selectedSensorQuery']
    middle_image = data['middleImage']
    platform = data['selectedPlatformQuery']

    if platform == 'Drone':
        # Do Drone Image query
        filtered_images = query_drone_images(data,data_root_dir)
    else:
        filtered_images = filter_images(geojson_features, year, experiment, location, 
                                        population, date, platform, sensor, middle_image)

    return jsonify(filtered_images)

@file_app.route('/dload_zipped', methods=['POST'])
def dload_zipped():
    data = request.get_json()
    geojson_features = data['geoJSON']
    year = data['selectedYearGCP']
    experiment = data['selectedExperimentGCP']
    location = data['selectedLocationGCP']
    population = data['selectedPopulationGCP']
    platform = data['selectedPlatformQuery']
    date = data['selectedDateQuery']
    sensor = data['selectedSensorQuery']
    middle_image = data['middleImage']

    filtered_images = filter_images(geojson_features, year, experiment, location,
                                    population, date, sensor, middle_image)

    # Move the images to a temporary directory
    temp_dir = '/tmp/filtered_images'

    # Clean up the temporary directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    # Create subdirectories for each label
    unique_labels = set([image['label'] for image in filtered_images])
    for label in unique_labels:
        os.makedirs(os.path.join(temp_dir, label), exist_ok=True)
    
    # Copy the images to the subdirectories
    for image in filtered_images:
        image_path = os.path.join(data_root_dir, 'Raw', year, experiment, location, population, 
                                  date, platform, sensor, image['imageName'])
        
        label_dir = os.path.join(temp_dir, image['label'])
        output_name = image['label'] + '_' + image['plot'] + '_' + image['imageName']
        output_path = os.path.join(label_dir, output_name)

        shutil.copy(image_path, output_path)
    
    # Zip the temporary directory
    shutil.make_archive('/tmp/filtered', 'zip', temp_dir)

    # Return the zipped file
    return send_file('/tmp/filtered.zip', as_attachment=True)

@file_app.route('/save_csv', methods=['POST'])
def save_csv():
    data = request.json
    selected_location_gcp = data.get('selectedLocationGcp')
    selected_population_gcp = data.get('selectedPopulationGcp')
    selected_year_gcp = data.get('selectedYearGcp')
    selected_experiment_gcp = data.get('selectedExperimentGcp')
    csv_data = data.get('csvData')
    filename = data.get('filename')

    # Construct file path based on GCP variables
    prefix = data_root_dir+'/Processed'
    file_path = os.path.join(prefix, selected_year_gcp, selected_experiment_gcp, selected_location_gcp, 
                             selected_population_gcp, filename)

    # Save CSV data to file
    pd.DataFrame(csv_data).to_csv(file_path, index=False)

    return jsonify({"status": "success", "message": "CSV data saved successfully"}), 200
    
@file_app.route('/save_geojson', methods=['POST'])
def save_geojson():
    data = request.json
    selected_location_gcp = data.get('selectedLocationGcp')
    selected_population_gcp = data.get('selectedPopulationGcp')
    selected_year_gcp = data.get('selectedYearGcp')
    selected_experiment_gcp = data.get('selectedExperimentGcp')
    geojson_data = data.get('geojsonData')
    filename = data.get('filename')

    # Load GeoJSON data into a GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(geojson_data)

    # Construct file path based on GCP variables
    prefix = data_root_dir+'/Intermediate'
    file_path = os.path.join(prefix, selected_year_gcp, selected_experiment_gcp, selected_location_gcp, 
                             selected_population_gcp, filename)
    
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save GeoDataFrame to file
    gdf.to_file(file_path, driver='GeoJSON')

    return jsonify({"status": "success", "message": "GeoJSON data saved successfully"})

@file_app.route('/load_geojson', methods=['POST'])
def load_geojson():
    data = request.json
    selected_location_gcp = data.get('selectedLocationGcp')
    selected_population_gcp = data.get('selectedPopulationGcp')
    selected_year_gcp = data.get('selectedYearGcp')
    selected_experiment_gcp = data.get('selectedExperimentGcp')
    filename = data.get('filename')

    print(data, flush=True)

    # Construct file path
    prefix = data_root_dir+'/Intermediate'
    file_path = os.path.join(prefix, selected_year_gcp, selected_experiment_gcp, selected_location_gcp, 
                             selected_population_gcp, filename)

    # Load GeoJSON data from file
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            geojson_data = json.load(file)
        return jsonify(geojson_data)
    else:
        return jsonify({"status": "error", "message": "File not found"})
    
@file_app.route('/get_odm_logs', methods=['GET'])
def get_odm_logs():
    logs_path = os.path.join(data_root_dir, 'temp', 'project', 'code', 'logs.txt')
    print("Logs Path:", logs_path)  # Debug statement
    if os.path.exists(logs_path):
        with open(logs_path, 'r') as log_file:
            lines = log_file.readlines()
            latest_logs = lines[-20:]  # Get the last 20 lines
        return jsonify({"log_content": ''.join(latest_logs)}), 200
    else:
        print("Logs not found at path:", logs_path)  # Debug statement
        return jsonify({"error": "Logs not found"}), 404

@file_app.route('/run_odm', methods=['POST'])
def run_odm_endpoint():
    data = request.json
    location = data.get('location')
    population = data.get('population')
    date = data.get('date')
    year = data.get('year')
    experiment = data.get('experiment')
    platform = data.get('platform')
    sensor = data.get('sensor')
    temp_dir = data.get('temp_dir')
    reconstruction_quality = data.get('reconstruction_quality')
    custom_options = data.get('custom_options')

    if not temp_dir:
        temp_dir = os.path.join(data_root_dir, 'temp/project')
    if not reconstruction_quality:
        reconstruction_quality = 'Low'

    # Run ODM
    args = make_odm_args(data_root_dir, 
                         location, 
                         population, 
                         date, 
                         year, 
                         experiment, 
                         platform, 
                         sensor, 
                         temp_dir, 
                         reconstruction_quality, 
                         custom_options)
    try:
        # Check if the container exists
        command = f"docker ps -a --filter name=GEMINI-Container --format '{{{{.Names}}}}'"
        command = command.split()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        container_exists = "GEMINI-Container" in stdout.decode().strip()

        if container_exists:
            print('Removing temp folder...')
            folder_to_delete = os.path.join(data_root_dir, 'temp', 'project')
            cleanup_command = f"docker exec GEMINI-Container rm -rf {folder_to_delete}"
            cleanup_command = cleanup_command.split()
            cleanup_process = subprocess.Popen(cleanup_command, stderr=subprocess.STDOUT)
            cleanup_process.wait()
            
            # Stop the container if it's running
            command = f"docker stop GEMINI-Container"
            command = command.split()
            process = subprocess.Popen(command, stderr=subprocess.STDOUT)
            process.wait()

            # Remove the container if it exists
            command = f"docker rm GEMINI-Container"
            command = command.split()
            process = subprocess.Popen(command, stderr=subprocess.STDOUT)
            process.wait()

        # # Proceed with the reset and starting threads
        # reset_thread = threading.Thread(target=reset_odm, args=(args,), daemon=True)
        # reset_thread.start()
        # reset_thread.join()  # Ensure reset thread is finished before proceeding
        
        # Run ODM in a separate thread
        thread = threading.Thread(target=run_odm, args=(args,), daemon=True)
        thread.start()
        
        # Run progress tracker
        logs_path = os.path.join(data_root_dir, 'temp/project/code/logs.txt')
        progress_file = os.path.join(data_root_dir, 'temp/progress.txt')
        os.makedirs(os.path.dirname(progress_file), exist_ok=True)
        thread_prog = threading.Thread(target=monitor_log_updates, args=(logs_path, progress_file), daemon=True)
        thread_prog.start()

        return jsonify({"status": "success", "message": "ODM processing started successfully"})
    except Exception as e:
        print('Error has occured: ', e)
        # Signal threads to stop
        stop_event = threading.Event()
        stop_event.set()
        
        # Optionally, wait for threads to finish if needed
        thread_prog.join()
        thread.join()
        return make_response(jsonify({"status": "error", "message": f"ODM processing failed to start {str(e)}"}), 404)


        
@file_app.route('/stop_odm', methods=['POST'])
def stop_odm():
    global data_root_dir
    try:
        print('ODM processed stopped by user.')
        print('Removing temp folder...')
        folder_to_delete = os.path.join(data_root_dir, 'temp', 'project')
        cleanup_command = f"docker exec GEMINI-Container rm -rf {folder_to_delete}"
        cleanup_command = cleanup_command.split()
        cleanup_process = subprocess.Popen(cleanup_command, stderr=subprocess.STDOUT)
        cleanup_process.wait()
        
        print('Stopping ODM process...')
        stop_event = threading.Event()
        stop_event.set()
        command = f"docker stop GEMINI-Container"
        command = command.split()
        # Run the command
        process = subprocess.Popen(command, stderr=subprocess.STDOUT)
        process.wait()
        
        print('Removing ODM container...')
        command = f"docker rm GEMINI-Container"
        command = command.split()
        # Run the command
        process = subprocess.Popen(command, stderr=subprocess.STDOUT)
        process.wait()
        # reset_odm(data_root_dir)
        shutil.rmtree(os.path.join(data_root_dir, 'temp'))
        return jsonify({"message": "ODM process stopped"}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": e.stderr.decode("utf-8")}), 500

@file_app.route('/get_ortho_progress', methods=['GET'])
def get_ortho_progress():
    return jsonify(latest_data)

@file_app.route('/get_ortho_metadata', methods=['GET'])
def get_ortho_metadata():
    global data_root_dir
    date = request.args.get('date')
    platform = request.args.get('platform')
    sensor = request.args.get('sensor')
    year = request.args.get('year')
    experiment = request.args.get('experiment')
    location = request.args.get('location')
    population = request.args.get('population')
    
    metadata_path = os.path.join(data_root_dir, 'Processed', year, experiment, location, population, date, platform, sensor, 'ortho_metadata.json')
    
    if not os.path.exists(metadata_path):
        return jsonify({"error": "Metadata file not found"}), 404
    
    try:
        with open(metadata_path, 'r') as file:
            metadata = json.load(file)
        return jsonify({
            "quality": metadata.get("quality", "N/A"),
            "timestamp": metadata.get("timestamp", "N/A")
        })
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON in metadata file"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
    
@file_app.route('/download_ortho', methods=['POST'])
def download_ortho():
    data = request.get_json()
    try:
        file_path = os.path.join(
            data_root_dir,
            'Processed',
            data['year'],
            data['experiment'],
            data['location'],
            data['population'],
            data['date'],
            data['platform'],
            data['sensor'],
            f"{data['date']}-RGB.png"
        )
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            try:
                print("Attempting to convert TIF to PNG...")
                
                # convert png to tif
                tif_path = file_path.replace('.png', '.tif')
                
                # convert tif to png
                if os.path.exists(tif_path):
                    convert_tif_to_png(tif_path)
                    
                else:
                    print(f"File not found: {tif_path}")
                    return jsonify({'error': f'File not found: {file_path}'}), 404
            except Exception as e:
                print(f"An error occurred while converting the file: {str(e)}")
                return jsonify({'error': str(e)}), 500

        # Returns the file as an attachment so that the browser downloads it.
        print(f"Sending file: {file_path}")
        return send_file(
            file_path,
            mimetype="image/png",
            as_attachment=True,
            download_name=os.path.basename(file_path)  # For Flask 2.0+
        )
        
    except Exception as e:
        print(f"An error occurred while downloading the ortho: {str(e)}")
        return jsonify({'error': str(e)}), 500
        

@file_app.route('/delete_ortho', methods=['POST'])
def delete_ortho():
    global data_root_dir
    data = request.json
    year = data.get('year')
    experiment = data.get('experiment')
    location = data.get('location')
    population = data.get('population')
    date = data.get('date')
    platform = data.get('platform')
    sensor = data.get('sensor')
    # modify when allowing for creation of orthos with same date and different quality
    ortho_path = os.path.join(data_root_dir, 'Processed', year, experiment, location, population, date)
    try:
        shutil.rmtree(ortho_path)
    except FileNotFoundError:
        print(f"Directory not found: {ortho_path}")
    except PermissionError:
        print(f"Permission denied: Unable to delete {ortho_path}")
    except Exception as e:
        print(f"An error occurred while deleting {ortho_path}: {str(e)}")
    return jsonify({"message": "Ortho file deleted successfully"}), 200

def update_progress_file(progress_file, progress, debug=False):
    with open(progress_file, 'w') as pf:
        pf.write(f"{progress}%")
        latest_data['ortho'] = progress
        if debug:
            print('Ortho progress updated:', progress)
               
def monitor_log_updates(logs_path, progress_file):
    
    try:
        progress_stages = [
            "Running dataset stage", # After spin up a docker container
            "Finished dataset stage", # After finish loading dataset
            "Computing pair matching", # After finish feature extraction, before the pair matching
            "Merging features onto tracks", # After pair matching, before merging features, 241.8s
            "Export reconstruction stats", # After Ceres Solver Report
            "Finished opensfm stage", # After Undistorting images
            "Densifying point-cloud completed", # After fusing depth maps
            "Finished openmvs stage", # After Finished openmvs stage, 31.83s
            "Finished odm_filterpoints stage", # Finished odm_filterpoints stage
            "Finished mvs_texturing stage", # After Finished mvs_texturing stage, 57.216s
            "Finished odm_georeferencing stage", # After Finished odm_georeferencing stage 
            "Finished odm_dem stage", # After Finished odm_dem stage
            "Finished odm_orthophoto stage", # After Finished odm_orthophoto stage
            "Finished odm_report stage",  # After Finished odm_report stage
            "Finished odm_postprocess stage", # Finished odm_postprocess stage
            "ODM app finished",             # ODM Processes are done, but some additional steps left
            "Copied RGB.tif",               # scripts/orthomosaic_generation.py L124
            "Generated DEM-Pyramid.tif",
            "Copied DEM.tif",
            "Generated RGB-Pyramid.tif",        # # scripts/orthomosaic_generation.py L163
        ]   
        
        with open(progress_file, 'w') as file:
            file.write("0")
        
        # Wait for the log file to be created
        while not os.path.exists(logs_path):
            print("Waiting for log file to be created...")
            time.sleep(5)  # Check every 5 seconds
        
        print("Log file found. Monitoring for updates.")
        
        # Log file exists, start monitoring
        with open(logs_path, 'r') as file:
            # Start by reading the file from the beginning
            file.seek(0)
            current_stage = -1
            while True:
                line = file.readline() # It will read the file line by line
                if line:
                    # print(line)
                    found_stages_msg = False
                    for idx, step in enumerate(progress_stages):
                        if step in line:
                            found_stages_msg = True
                            current_stage = idx
                            break

                    if found_stages_msg and current_stage > -1:
                        current_progress = (current_stage+1) / len(progress_stages) * 100
                        update_progress_file(progress_file, round(current_progress))
                        print(progress_stages[current_stage])
                        print(f"Progress updated: {current_progress:.1f}%")
                        if current_stage == len(progress_stages) - 1:
                            break
                else:
                    time.sleep(10)  # Sleep briefly to avoid busy waiting

    except Exception as e:
        # Handle exception: log it, set a flag, etc.
        print(f"Error in thread: {e}")
        
### CVAT #### 
@file_app.route('/start_cvat', methods=['POST'])
def start_cvat():
    global data_root_dir
    clone_dir = os.path.join(data_root_dir, 'cvat')
    
    # Create the directory if it doesn't exist
    os.makedirs(clone_dir, exist_ok=True)
    
    # Define the path for the compose directory
    compose_dir = os.path.join(clone_dir, 'cvat')

    try:
        # Get a list of all containers with 'cvat' or 'traefik' in their name
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=cvat", "--format", "{{.Names}}"],
            stdout=subprocess.PIPE
        )
        cvat_containers = result.stdout.decode('utf-8').strip().split('\n')
        cvat_containers = [container for container in cvat_containers if container]  # filter out empty strings

        # Add traefik to the list of containers to restart
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=traefik", "--format", "{{.Names}}"],
            stdout=subprocess.PIPE
        )
        traefik_containers = result.stdout.decode('utf-8').strip().split('\n')
        traefik_containers = [container for container in traefik_containers if container]  # filter out empty strings

        # Combine the lists of containers
        all_containers = cvat_containers + traefik_containers

    except Exception as e:
        return jsonify({"error": f"Error checking for CVAT or Traefik containers: {str(e)}"}), 404

    try:
        if all_containers:
            # Restart each container with 'cvat' or 'traefik' in its name
            print(f"Restarting containers: {all_containers}")
            for container in all_containers:
                subprocess.run(["docker", "restart", container])
            print("All relevant containers have been restarted.")
        else:
            # Clone the repository if needed and run docker-compose
            if not os.path.exists(compose_dir):
                subprocess.run(
                    ["git", "clone", "https://github.com/cvat-ai/cvat"], cwd=clone_dir
                )
                
            # Check if docker-compose.yml exists before starting docker-compose
            compose_file = os.path.join(compose_dir, 'docker-compose.yml')
            if not os.path.exists(compose_file):
                return jsonify({"error": "docker-compose.yml not found in the cloned repository"}), 404
            
            # Start CVAT with docker-compose
            subprocess.run(
                ["docker-compose", "up", "-d"], cwd=compose_dir
            )
            print("Starting CVAT container with docker-compose...")

        # Wait for specific services to be fully up and running
        services_to_check = ["cvat_server", "cvat_ui"]
        max_retries = 30
        for i in range(max_retries):
            # Check the status of the containers
            result = subprocess.run(
                ["docker-compose", "ps", "--services", "--filter", "status=running"], cwd=compose_dir, stdout=subprocess.PIPE
            )
            running_services = result.stdout.decode('utf-8').strip().split('\n')

            # Check if all expected services are running
            if all(service in running_services for service in services_to_check):
                print("All required CVAT services are running.")
                break

            print(f"Waiting for CVAT services to start... ({i + 1}/{max_retries})")
            time.sleep(5)  # Wait 5 seconds before checking again

        if i == max_retries - 1:
            return jsonify({"error": "CVAT services failed to start in time"}), 500

        # Create superuser via docker exec
        subprocess.run(
            ["docker", "exec", "-it", "cvat_server", "bash", "-c", "'python3 ~/manage.py createsuperuser'"]
        )
        print("CVAT superuser created.")

        # Poll the CVAT server to check if it's running
        cvat_url = "http://localhost:8080/api/server/about"
        for i in range(max_retries):
            try:
                response = requests.get(cvat_url)
                if response.status_code == 200:
                    print("CVAT server is up and running.")
                    break
            except requests.exceptions.RequestException:
                pass  # Server is not ready yet

            print(f"Waiting for CVAT to start... ({i + 1}/{max_retries})")
            time.sleep(5)  # Wait 5 seconds before checking again

        # If the server didn't start within the max retries, return an error
        if i == max_retries - 1:
            return jsonify({"error": "CVAT server failed to start in time"}), 500

        return jsonify({"status": "CVAT and Traefik containers restarted and superuser created"})
    except Exception as e:
        print(f"Error starting CVAT or Traefik containers: {str(e)}")
        return jsonify({"error": f"Error starting CVAT or Traefik containers: {str(e)}"}), 404

### ROVER LABELS PREPARATION ###
@file_app.route('/check_labels/<path:dir_path>', methods=['GET'])
def check_labels(dir_path):
    global data_root_dir
    data = []
    
    # get labels path
    labels_path = Path(data_root_dir)/dir_path

    if labels_path.exists() and labels_path.is_dir():
        # Use glob to find all .txt files in the directory
        txt_files = list(labels_path.glob('*.txt'))
        
        # Check if there are more than one .txt files
        if len(txt_files) > 1:
            data.append(str(labels_path))

    return jsonify(data)

@file_app.route('/check_existing_labels', methods=['POST'])
def check_existing_labels():
    global data_root_dir
    
    data = request.json
    fileList = data['fileList']
    dirPath = data['dirPath']
    full_dir_path = os.path.join(data_root_dir, dirPath)

    # existing_files = set(os.listdir(full_dir_path)) if os.path.exists(full_dir_path) else set()
    existing_files = [file.name for file in Path(full_dir_path).rglob('*.txt')]
    new_files = [file for file in fileList if file not in existing_files]

    print(f"Uploading {str(len(new_files))} out of {str(len(fileList))} files to {dirPath}")

    return jsonify(new_files), 200

@file_app.route('/upload_trait_labels', methods=['POST'])
def upload_trait_labels():
    global data_root_dir
    
    dir_path = request.form.get('dirPath')
    full_dir_path = os.path.join(data_root_dir, dir_path)
    os.makedirs(full_dir_path, exist_ok=True)

    for file in request.files.getlist("files"):
        filename = secure_filename(file.filename)
        file_path = os.path.join(full_dir_path, filename)
        print(f'Saving {file_path}...')
        file.save(file_path)

    return jsonify({'message': 'Files uploaded successfully'}), 200

def split_data(labels, images, test_size=0.2):
    # Calculate split index
    split_index = int(len(labels) * (1 - test_size))
    
    # Split the labels and images into train and validation sets
    labels_train = labels[:split_index]
    labels_val = labels[split_index:]
    
    images_train = images[:split_index]
    images_val = images[split_index:]
    
    return labels_train, labels_val, images_train, images_val

def copy_files_to_folder(source_files, target_folder):
    for source_file in source_files:
        target_file = target_folder / source_file.name
        if not target_file.exists():
            shutil.copy(source_file, target_file)
            
def remove_files_from_folder(folder):
    for file in folder.iterdir():
        if file.is_file():
            file.unlink()

def prepare_labels(annotations, images_path):
    
    try:
        global data_root_dir, labels_train_folder, labels_val_folder, images_train_folder, images_val_folder
        
        # path to labels
        labels_train_folder = annotations.parent/'labels'/'train'
        labels_val_folder = annotations.parent/'labels'/'val'
        images_train_folder = annotations.parent/'images'/'train'
        images_val_folder = annotations.parent/'images'/'val'
        labels_train_folder.mkdir(parents=True, exist_ok=True)
        labels_val_folder.mkdir(parents=True, exist_ok=True)
        images_train_folder.mkdir(parents=True, exist_ok=True)
        images_val_folder.mkdir(parents=True, exist_ok=True)

        # obtain path to images
        images = list(images_path.rglob('*.jpg')) + list(images_path.rglob('*.png'))
        
        # split images to train and val
        labels = list(annotations.glob('*.txt'))
        label_stems = set(Path(label).stem for label in labels)
        filtered_images = [image for image in images if Path(image).stem in label_stems]
        labels_train, labels_val, images_train, images_val = split_data(labels, filtered_images)

        # link images and labels to folder
        copy_files_to_folder(labels_train, labels_train_folder)
        copy_files_to_folder(labels_val, labels_val_folder)
        copy_files_to_folder(images_train, images_train_folder)
        copy_files_to_folder(images_val, images_val_folder)
        
        # check if images_train_folder and images_val_folder are not empty
        if not any(images_train_folder.iterdir()) or not any(images_val_folder.iterdir()):
            return False
        else:
            return True
        
    except Exception as e:
        print(f'Error preparing labels for training: {e}')

### ROVER MODEL TRAINING ###
def check_model_details(key, value = None):
    
    # get base folder, args file and results file
    base_path = key.parent.parent
    args_file = base_path / 'args.yaml'
    results_file = base_path / 'results.csv'
    
    # get epochs, batch size and image size
    values = []
    with open(args_file, 'r') as file:
        args = yaml.safe_load(file)
        epochs = args.get('epochs')
        batch = args.get('batch')
        imgsz = args.get('imgsz')
        
        values.extend([epochs, batch, imgsz])
    
    # get mAP of model
    df = pd.read_csv(results_file, delimiter=',\s+', engine='python')
    mAP = round(df['metrics/mAP50(B)'].iloc[-1], 2)  # Get the last value in the column
    values.extend([mAP])
    
    # get run name
    run = base_path.name
    match = re.search(r'-([A-Za-z0-9]+)$', run)
    id = match.group(1)
    
    # get date(s)
    if value is not None:
        date = ', '.join(value)
    else:
        date = None
    
    # get platform
    platform = base_path.parts[-3]
    
    # get sensor
    sensor = base_path.parts[-2].split()[0]
    
    # collate details
    details = {'id': id, 'dates': date, 'platform': platform, 'sensor': sensor, 'epochs': epochs, 'batch': batch, 'imgsz': imgsz, 'map': mAP}
    
    return details

@file_app.route('/get_model_info', methods=['POST'])
def get_model_info():
    data = request.json
    details_data = []
    
    # iterate through each existing model
    for key in data:
        details = check_model_details(Path(key), value = data[key])
        details_data.append(details)

    return jsonify(details_data)

def get_labels(labels_path):
    unique_labels = set()

    # Iterate over the files in the directory
    for filename in os.listdir(labels_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(labels_path, filename)
            with open(file_path, 'r') as file:
                for line in file:
                    label = line.split()[0]  # Extracting the label
                    unique_labels.add(label)

    sorted_unique_labels = sorted(unique_labels, key=lambda x: int(x))
    return list(sorted_unique_labels)

def scan_for_new_folders(save_path):
    global latest_data, training_stopped_event, new_folder, results_file
    known_folders = {os.path.join(save_path, f) for f in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, f))}

    while not training_stopped_event.is_set():  # Continue while training is ongoing
        # Check for new folders
        for folder_name in os.listdir(save_path):
            folder_path = os.path.join(save_path, folder_name)
            if os.path.isdir(folder_path) and folder_path not in known_folders:
                known_folders.add(folder_path)  # Add new folder to the set
                new_folder = folder_path  # Update global variable
                results_file = os.path.join(folder_path, 'results.csv')

                # Continuously check results.csv for updates
                while not os.path.isfile(results_file):
                    time.sleep(5)  # Check every 5 seconds

                # Periodically read results.csv for updates
                while os.path.exists(results_file) and os.path.isfile(results_file):
                    try:
                        df = pd.read_csv(results_file, delimiter=',\s+', engine='python')
                        latest_data['epoch'] = int(df['epoch'].iloc[-1])  # Update latest epoch
                        latest_data['map'] = df['metrics/mAP50(B)'].iloc[-1]  # Update latest mAP
                    except Exception as e:
                        print(f"Error reading {results_file}: {e}")
                    time.sleep(5)  # Update every 30 seconds

        time.sleep(5)  # Check for new folders every 10 seconds
        
@file_app.route('/get_progress', methods=['GET'])
def get_training_progress():
    for key, value in latest_data.items():
        if isinstance(value, np.int64):
            latest_data[key] = int(value)
    print(latest_data)
    return jsonify(latest_data)

@file_app.route('/train_model', methods=['POST'])
def train_model():
    global data_root_dir, latest_data, training_stopped_event, new_folder, train_labels, training_process
    
    try:
        # receive the parameters
        epochs = int(request.json['epochs'])
        batch_size = int(request.json['batchSize'])
        image_size = int(request.json['imageSize'])
        location = request.json['location']
        population = request.json['population']
        date = request.json['date']
        trait = request.json['trait']
        sensor = request.json['sensor']
        platform = request.json['platform']
        year = request.json['year']
        experiment = request.json['experiment']
        
        # prepare labels
        annotations = Path(data_root_dir)/'Intermediate'/year/experiment/location/population/date/platform/sensor/f'Labels/{trait} Detection/annotations'
        all_images = Path(data_root_dir)/'Raw'/year/experiment/location/population/date/platform/sensor/'Images'
        # all_images = Path('/home/gemini/mnt/d/Annotations/Plant Detection/obj_train_data')
        check_if_images_exist = prepare_labels(annotations, all_images)
        # wait for 1 minute
        time.sleep(60)
        if check_if_images_exist == False:
            return jsonify({"error": "No images found for training. Press stop and upload images."}), 404
        
        # extract labels
        labels_path = Path(data_root_dir)/'Intermediate'/year/experiment/location/population/date/platform/sensor/f'Labels/{trait} Detection/labels/train'
        labels = get_labels(labels_path)
        labels_arg = " ".join(labels).split()
        
        # other training args
        pretrained = "yolov8n.pt"
        save_train_model = Path(data_root_dir)/'Intermediate'/year/experiment/location/population/'Training'/platform
        scan_save = Path(data_root_dir)/'Intermediate'/year/experiment/location/population/'Training'/platform/f'{sensor} {trait} Detection'
        scan_save = Path(scan_save)
        scan_save.mkdir(parents=True, exist_ok=True)
        latest_data['epoch'] = 0
        latest_data['map'] = 0
        training_stopped_event.clear()
        threading.Thread(target=scan_for_new_folders, args=(scan_save,), daemon=True).start()
        images = Path(data_root_dir)/'Intermediate'/year/experiment/location/population/date/platform/sensor/f'Labels/{trait} Detection'
        
        cmd = (
            f"python {TRAIN_MODEL} "
            f"--pretrained '{pretrained}' "
            f"--images '{images}' "
            f"--save '{save_train_model}' "
            f"--sensor '{sensor}' "
            f"--date '{date}' "
            f"--trait '{trait}' "
            f"--image-size '{image_size}' "
            f"--epochs '{epochs}' "
            f"--batch-size {batch_size} "
            f"--labels {' '.join(labels_arg)} "
        )
        print(cmd)
        
        training_process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        threading.Thread(target=stream_output, args=(training_process,), daemon=True).start()
        time.sleep(5)  # Wait for 5 seconds
        if training_process.poll() is None:
            print("Process started successfully and is running.")
        else:
            print("Process failed to start or exited immediately.")
        
        return jsonify({"message": "Training started"}), 202

    except subprocess.CalledProcessError as e:
        error_output = e.stderr.decode('utf-8')
        return jsonify({"error": error_output}), 500
    
@file_app.route('/stop_training', methods=['POST'])
def stop_training():
    global training_stopped_event, new_folder, labels_train_folder, labels_val_folder, images_train_folder, images_val_folder, training_process

    try:        
        # stop training
        print('Training stopped by user.')
        if training_process is not None:
            training_process.terminate()
            training_process.wait()  # Optionally wait for the process to terminate
            print("Training process terminated.")
            training_process = None
        else:
            print("No training process running.")
            
        training_stopped_event.set()
        subprocess.run(f"rm -rf '{new_folder}'", check=True, shell=True)
        
        # unlink files
        remove_files_from_folder(labels_train_folder)
        remove_files_from_folder(labels_val_folder)
        remove_files_from_folder(images_train_folder)
        remove_files_from_folder(images_val_folder)
        
        return jsonify({"message": "Python process in container successfully stopped"}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": e.stderr.decode("utf-8")}), 500
    
@file_app.route('/done_training', methods=['POST'])
def done_training():
    global training_stopped_event, new_folder, results_file, labels_train_folder, labels_val_folder, images_train_folder, images_val_folder, training_process
    # container_name = 'train'
    try:
        # stop training
        print('Training stopped by user.')
        # kill_cmd = f"docker exec {container_name} pkill -9 -f python"
        # subprocess.run(kill_cmd, shell=True)
        # print(f"Sent SIGKILL to Python process in {container_name} container.")
        if training_process is not None:
            training_process.terminate()
            training_process.wait()  # Optionally wait for the process to terminate
            print("Training process terminated.")
            training_process = None
        else:
            print("No training process running.")
            
        # subprocess.run(f"rm -rf '{new_folder}'", check=True, shell=True)
        # print(f"Removed {new_folder}")
        
        # unlink files
        remove_files_from_folder(labels_train_folder)
        remove_files_from_folder(labels_val_folder)
        remove_files_from_folder(images_train_folder)
        remove_files_from_folder(images_val_folder)
        training_stopped_event.set()
        results_file = ''
        return jsonify({"message": "Python process in container successfully stopped"}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": e.stderr.decode("utf-8")}), 500

### ROVER LOCATE PLANTS ###
def check_locate_details(key):
    
    # get base folder, args file and results file
    base_path = key.parent
    results_file = base_path / 'locate.csv'
    
    # get run name
    run = base_path.name
    match = re.search(r'-([A-Za-z0-9]+)$', run)
    id = match.group(1)
    
    # get model id
    with open(base_path/'logs.yaml', 'r') as file:
        data = yaml.safe_load(file)
    model_id = data['model']
    
    
    # get stand count
    df = pd.read_csv(results_file)
    stand_count = len(df)
    
    # get date
    date = data['date']
    
    # get platform
    platform = base_path.parts[-4]
    
    # get sensor
    sensor = base_path.parts[-3]
    
    # get mAP of model
    date_index = base_path.parts.index(date[0]) if date[0] in base_path.parts else None
    if date_index and date_index > 0:
        # Construct a new path from the parts up to the folder before the known date
        root_path = Path(*base_path.parts[:date_index])
        results_file = root_path / 'Training' / platform / f'{sensor} Plant Detection' / f'Plant-{model_id[0]}' / 'results.csv'
    df = pd.read_csv(results_file, delimiter=',\s+', engine='python')
    mAP = round(df['metrics/mAP50(B)'].iloc[-1], 2)
    # values.extend([mAP])
    
    # collate details
    details = {'id': id, 'model': model_id, 'count': stand_count, 'date': date, 'platform': platform, 'sensor': sensor, 'performance': mAP}
    
    return details

@file_app.route('/get_locate_info', methods=['POST'])
def get_locate_info():
    data = request.json
    details_data = []
    
    # iterate through each existing model
    for key in data:
        details = check_locate_details(Path(key))
        details_data.append(details)
        
    return jsonify(details_data)

@file_app.route('/get_locate_progress', methods=['GET'])
def get_locate_progress():
    global save_locate
    txt_file = save_locate/'locate_progress.txt'
    
    # Check if the file exists
    if os.path.exists(txt_file):
        with open(txt_file, 'r') as file:
            number = file.read().strip()
            latest_data['locate'] = int(number)
        return jsonify(latest_data)
    else:
        return jsonify({'error': 'Locate progress not found'}), 404

def generate_hash(trait, length=6):
    """Generate a hash for model where it starts with the trait followed by a random string of characters.

    Args:
        trait (str): trait to be analyzed (plant, flower, pod, etc.)
        length (int, optional): Length for random sequence. Defaults to 5.
    """
    random_sequence = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    hash_id = f"{trait}-{random_sequence}"
    return hash_id

@file_app.route('/locate_plants', methods=['POST'])
def locate_plants():
    global data_root_dir, save_locate, locate_process
    
    # recieve parameters
    batch_size = int(request.json['batchSize'])
    location = request.json['location']
    population = request.json['population']
    date = request.json['date']
    platform = request.json['platform']
    sensor = request.json['sensor']
    year = request.json['year']
    experiment = request.json['experiment']
    model = request.json['model']
    id = request.json['id']
    
    # other args
    # container_dir = Path('/app/mnt/GEMINI-App-Data')
    images = Path(data_root_dir)/'Raw'/year/experiment/location/population/date/platform/sensor/'Images'
    disparity = Path(data_root_dir)/'Raw'/year/experiment/location/population/date/platform/sensor/'Disparity'
    configs = Path(data_root_dir)/'Raw'/year/experiment/location/population/date/platform/sensor/'Metadata'
    plotmap = Path(data_root_dir)/'Intermediate'/year/experiment/location/population/'Plot-Boundary-WGS84.geojson'
    
    # generate save folder
    version = generate_hash(trait='Locate')
    save_base = Path(data_root_dir)/'Intermediate'/year/experiment/location/population/date/platform/sensor/f'Locate'
    while (save_base / f'{version}').exists():
        version = generate_hash(trait='Locate')
    save_locate = Path(data_root_dir)/'Intermediate'/year/experiment/location/population/date/platform/sensor/f'Locate'/f'{version}'
    save_locate.mkdir(parents=True, exist_ok=True)
    model_path = Path(data_root_dir)/'Intermediate'/year/experiment/location/population/'Training'/f'{platform}'/'RGB Plant Detection'/f'Plant-{id}'/'weights'/'last.pt' # TODO: DEBUG
    # model_path = "/mnt/d/GEMINI-App-Data/Intermediate/2022/GEMINI/Davis/Legumes/Training/Amiga-Onboard/RGB Plant Detection/Plant-btRN26/weights/last.pt"
    
    # save logs file
    data = {"model": [id], "date": [date]}
    with open(save_locate/"logs.yaml", "w") as file:
        yaml.dump(data, file, default_flow_style=False)
        
    # create progress file
    with open(save_locate/"locate_progress.txt", "w") as file:
        pass
    
    # run locate
    cmd = (
        f"python -W ignore {LOCATE_PLANTS} "
        f"--images '{images}' --metadata '{configs}' --plotmap '{plotmap}' "
        f"--batch-size '{batch_size}' --model '{model_path}' --save '{save_locate}'"
    )

    if disparity.exists():
        cmd += " --skip-stereo"

    try:
        locate_process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        threading.Thread(target=stream_output, args=(locate_process,), daemon=True).start()
        time.sleep(5)  # Wait for 5 seconds
        if locate_process.poll() is None:
            print("Locate process started successfully and is running.")
            return jsonify({"message": "Locate started"}), 202
        else:
            print("Locate process failed to start or exited immediately.")
            return jsonify({"error": "Failed to start locate process." }), 404
    
    except subprocess.CalledProcessError as e:
        
        error_output = e.stderr.decode('utf-8')
        return jsonify({"error": error_output}), 404
    
@file_app.route('/stop_locate', methods=['POST'])
def stop_locate():
    global save_locate, locate_process
    
    try:
        print('Locate stopped by user.')
        if locate_process is not None:
            locate_process.terminate()
            locate_process.wait()
            print("Locate process terminated.")
            locate_process = None
        else:
            print("No locate process running.")

        subprocess.run(f"rm -rf '{save_locate}'", check=True, shell=True)
        return jsonify({"message": "Python process in container successfully stopped"}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": e.stderr.decode("utf-8")}), 500
    
### ROVER EXTRACT PLANTS ###
def update_or_add_entry(data, key, new_values):
    if key in data:
        # Update existing entry
        data[key].update(new_values)
    else:
        # Add new entry
        data[key] = new_values
        
@file_app.route('/get_extract_progress', methods=['GET'])
def get_extract_progress():
    global save_extract
    txt_file = save_extract/'extract_progress.txt'
    
    # Check if the file exists
    if os.path.exists(txt_file):
        with open(txt_file, 'r') as file:
            number = file.read().strip()
            latest_data['extract'] = int(number)
        return jsonify(latest_data)
    else:
        return jsonify({'error': 'Locate progress not found'}), 404
    
@file_app.route('/extract_traits', methods=['POST'])
def extract_traits():
    global data_root_dir, save_extract, temp_extract, model_id, summary_date, locate_id, trait_extract, extract_process
    
    try:
        # recieve parameters
        summary = request.json['summary']
        batch_size = int(request.json['batchSize'])
        model = request.json['model']
        trait = request.json['trait']
        trait_extract = request.json['trait']
        date = request.json['date']
        year = request.json['year']
        experiment = request.json['experiment']
        location = request.json['location']
        population = request.json['population']
        platform = request.json['platform']
        sensor = request.json['sensor']
        
        # extract model and summary information
        pattern = r"/[^/]+-([\w]+?)/weights"
        date_pattern = r"\b\d{4}-\d{2}-\d{2}\b"
        locate_pattern = r"Locate-(\w+)/"
        match = re.search(pattern, str(model))
        match_date = re.search(date_pattern, str(summary))
        match_locate_id = re.search(locate_pattern, str(summary))
        model_id = match.group(1)
        summary_date = match_date.group()
        locate_id = match_locate_id.group(1)
        
        # other args
        summary_path = Path(data_root_dir)/'Intermediate'/year/experiment/location/population/summary_date/platform/sensor/'Locate'/f'Locate-{locate_id}'/'locate.csv'
        model_path = Path(data_root_dir)/'Intermediate'/year/experiment/location/population/'Training'/platform/f'RGB {trait} Detection'/f'{trait}-{model_id}'/'weights'/'last.pt'
        images = Path(data_root_dir)/'Raw'/year/experiment/location/population/date/platform/sensor/'Images'
        disparity = Path(data_root_dir)/'Raw'/year/experiment/location/population/date/platform/sensor/'Disparity'
        plotmap = Path(data_root_dir)/'Intermediate'/year/experiment/location/population/'Plot-Boundary-WGS84.geojson'
        metadata = Path(data_root_dir)/'Raw'/year/experiment/location/population/date/platform/sensor/'Metadata'
        save = Path(data_root_dir)/'Processed'/year/experiment/location/population/date/platform/sensor/f'{date}-{platform}-{sensor}-Traits-WGS84.geojson'
        save_extract = Path(data_root_dir)/'Processed'/year/experiment/location/population/date/platform/sensor
        temp = Path(data_root_dir)/'Processed'/year/experiment/location/population/date/platform/sensor/'temp'
        temp_extract = Path(data_root_dir)/'Processed'/year/experiment/location/population/date/platform/sensor/'temp'
        temp_extract.mkdir(parents=True, exist_ok=True) #if it doesnt exists
        save_extract.mkdir(parents=True, exist_ok=True)
        
        # reset extract process (or initialize)
        extract_process = None
        
        # check if date is emerging
        emerging = date in summary
        
        # check if metadata path exists OR contains files
        if not metadata.exists() or not any(metadata.iterdir()):
            return jsonify({"error": "Platform logs not found or empty. Please press stop and upload necessary logs."}), 404
        
        # run extract
        if emerging:
            if disparity.exists():
                cmd = (
                    f"python -W ignore {EXTRACT_TRAITS} "
                    f"--emerging --summary '{summary_path}' --images '{images}' --plotmap '{plotmap}' "
                    f"--batch-size {batch_size} --model-path '{model_path}' --save '{save_extract}' "
                    f"--metadata '{metadata}' --temp '{temp}' --trait '{trait}' --skip-stereo --geojson-filename '{save}'"
                )
            else:
                cmd = (
                    f"python -W ignore {EXTRACT_TRAITS} "
                    f"--emerging --summary '{summary_path}' --images '{images}' --plotmap '{plotmap}' "
                    f"--batch-size {batch_size} --model-path '{model_path}' --save '{save}' "
                    f"--metadata '{metadata}' --temp '{temp}' --trait '{trait}' --geojson-filename '{save}'"
                )
        else:
            if disparity.exists():
                cmd = (
                    f"python -W ignore {EXTRACT_TRAITS} "
                    f"--summary '{summary_path}' --images '{images}' --plotmap '{plotmap}' "
                    f"--batch-size {batch_size} --model-path '{model_path}' --save '{save}' "
                    f"--metadata '{metadata}' --temp '{temp}' --trait '{trait}' --skip-stereo --geojson-filename '{save}'"
                )
            else:
                cmd = (
                    f"python -W ignore {EXTRACT_TRAITS} "
                    f"--summary '{summary_path}' --images '{images}' --plotmap '{plotmap}' "
                    f"--batch-size {batch_size} --model-path '{model_path}' --save '{save}' "
                    f"--metadata '{metadata}' --temp '{temp}' --trait '{trait}' --geojson-filename '{save}'"
                )
        print(cmd)
        
        extract_process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        threading.Thread(target=stream_output, args=(extract_process,), daemon=True).start()
        time.sleep(5)  # Wait for 5 seconds
        if extract_process.poll() is None:
            print("Extract process started successfully and is running.")
            return jsonify({"message": "Extract started"}), 202
        else:
            print("Extract process failed to start or exited immediately.")
            return jsonify({"error": 
                "Failed to start extraction process. Check if you have corectly uploaded images/metadata"}), 404
    
    except subprocess.CalledProcessError as e:
        error_output = e.stderr.decode('utf-8')
        return jsonify({"status": "error", "message": str(error_output)}), 404
    
@file_app.route('/stop_extract', methods=['POST'])
def stop_extract():
    global save_extract, temp_extract, extract_process
    try:
        print('Extract stopped by user.')
        if extract_process is not None:
            extract_process.terminate()
            extract_process.wait()
            print("Extract process terminated.")
            extract_process = None
        else:
            print("No extract process running.")
        
        subprocess.run(f"rm -rf '{save_extract}/logs.yaml'", check=True, shell=True)
        subprocess.run(f"rm -rf '{temp_extract}'", check=True, shell=True)
        return jsonify({"message": "Python process successfully stopped"}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": e.stderr.decode("utf-8")}), 500

def update_plot_index(directory, image_name, plot_index, position, camera):
    # The directory from frontend is .../RGB/Images/<camera>
    # The CSV is in .../RGB/Metadata
    metadata_dir = os.path.abspath(os.path.join(directory, '..', '..', 'Metadata'))
    csv_path = os.path.join(metadata_dir, 'msgs_synced.csv')

    if not os.path.exists(csv_path):
        return jsonify({"error": f"msgs_synced.csv not found at {csv_path}"}), 404

    df = pd.read_csv(csv_path)

    if 'plot_index' not in df.columns:
        df['plot_index'] = -1

    image_column = f'/{camera}/rgb_file'
    if image_column not in df.columns:
        return jsonify({"error": f"Image column '{image_column}' not found in {csv_path}"}), 404

    try:
        row_index = df.index[df[image_column] == image_name].tolist()[0]
    except IndexError:
        return jsonify({"error": f"Image '{image_name}' not found in column '{image_column}'"}), 404

    current_plot_index = int(plot_index)

    if position == 'start':
        df.loc[row_index, 'plot_index'] = current_plot_index
    elif position == 'end':
        # Mark all rows from start to end
        start_indices = df.index[df['plot_index'] == current_plot_index].tolist()
        if not start_indices:
            # If start not found, just mark the end row.
            df.loc[row_index, 'plot_index'] = current_plot_index
        else:
            start_index = start_indices[0]
            # Ensure end is after start
            if row_index >= start_index:
                df.loc[start_index:row_index, 'plot_index'] = current_plot_index
            else:
                # If end is before start, just mark the end row
                 df.loc[row_index, 'plot_index'] = current_plot_index


    df.to_csv(csv_path, index=False)
    return jsonify({"status": "success"})

@file_app.route('/mark_plot_start', methods=['POST'])
def mark_plot_start():
    data = request.json
    return update_plot_index(data['directory'], data['image_name'], data['plot_index'], 'start', data['camera'])

@file_app.route('/mark_plot_end', methods=['POST'])
def mark_plot_end():
    data = request.json
    return update_plot_index(data['directory'], data['image_name'], data['plot_index'], 'end', data['camera'])

@file_app.route('/get_plot_data', methods=['POST'])
def get_plot_data():
    data = request.json
    directory = data['directory']
    image_name = data['image_name']
    camera = data['camera']

    csv_path = os.path.join(directory, 'msgs_synced.csv')
    if not os.path.exists(csv_path):
        return jsonify({'error': 'msgs_synced.csv not found'}), 404

    df = pd.read_csv(csv_path)

    # Construct the image column name based on the camera
    image_column = f'/{camera}/rgb_file'

    if image_column not in df.columns:
        return jsonify({'error': f'Column {image_column} not found in {csv_path}'}), 400

    try:
        # Find the row that matches the image name
        row = df[df[image_column] == image_name]
        if row.empty:
            return jsonify({'error': f'Image {image_name} not found in {image_column}'}), 404
        
        # Extract the plot index
        plot_index = row['plot_index'].values[0]

        return jsonify({'plot_index': plot_index})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@file_app.route('/done_extract', methods=['POST'])
def done_extract():
    global temp_extract, save_extract, model_id, summary_date, locate_id, trait_extract, extract_process
    try:
        # update logs file
        logs_file = Path(save_extract)/'logs.yaml'
        if logs_file.exists():
            with open(logs_file, 'r') as file:
                data = yaml.safe_load(file) or {} # use an empty dict if the file is empty
        else:
            data = {}
        new_values = {
            "model": model_id,
            "locate": summary_date,
            "id": locate_id
        }
        update_or_add_entry(data, trait_extract, new_values)
        with open(logs_file, 'w') as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)
        
        print('Extract stopped by user.')
        if extract_process is not None:
            extract_process.terminate()
            extract_process.wait()
            print("Extract process terminated.")
            extract_process = None
        else:
            print("No extract process running.")
        subprocess.run(f"rm -rf '{temp_extract}'", check=True, shell=True)
        return jsonify({"message": "Python process in container successfully stopped"}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": e.stderr.decode("utf-8")}), 500

# FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Flask app to FastAPI
app.mount("/flask_app", WSGIMiddleware(file_app))

# Add Titiler to FastAPI
# app.mount("/cog", app=titiler_app, name='titiler')

if __name__ == "__main__":
    
    # Add arguments to the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir', type=str, default='~/GEMINI-App-Data',required=False)
    parser.add_argument('--flask_port', type=int, default=5000,required=False) # Default port is 5000
    parser.add_argument('--titiler_port', type=int, default=8091,required=False) # Default port is 8091
    args = parser.parse_args()

    # Print the arguments to the console
    print(f"flask_port: {args.flask_port}")
    print(f"titiler_port: {args.titiler_port}")

    # Update global data_root_dir from the argument
    global data_root_dir
    data_root_dir = args.data_root_dir
    if "~" in data_root_dir:
        data_root_dir = os.path.expanduser(data_root_dir)
    print(f"data_root_dir: {data_root_dir}")

    UPLOAD_BASE_DIR = os.path.join(data_root_dir, 'Raw')

    global now_drone_processing
    now_drone_processing = False

    # Start the Titiler server using the subprocess module
    titiler_command = f"uvicorn titiler.application.main:app --reload --port {args.titiler_port}"
    titiler_process = subprocess.Popen(titiler_command, shell=True)

    # Start the Flask server
    uvicorn.run(app, host="127.0.0.1", port=args.flask_port)

    # Terminate the Titiler server when the Flask server is shut down
    titiler_process.terminate()