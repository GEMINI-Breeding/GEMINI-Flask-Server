# Standard library imports
import os
import re
import subprocess
import threading
import time
import glob
import yaml
import csv
import shutil
import traceback
import tempfile
import torch
import argparse
import requests
from multiprocessing import active_children, Process
from pathlib import Path

import unicodedata
import re

# Third-party library imports

import uvicorn
import json
import pandas as pd
import geopandas as gpd
import numpy as np
from flask import Flask, make_response, send_from_directory, jsonify, request, send_file

# Import inference module
from scripts.roboflow_inference import register_inference_routes
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
from scripts.orthomosaic_generation import run_odm, reset_odm, make_odm_args, convert_tif_to_png, monitor_log_updates
from scripts.utils import process_directories_in_parallel, process_directories_in_parallel_from_db, stream_output
from scripts.utils import update_or_add_entry, split_data, prepare_labels, remove_files_from_folder, copy_files_to_folder, check_model_details
from scripts.utils import generate_hash
from scripts.gcp_picker import collect_gcp_candidate, process_exif_data_async, refresh_gcp_candidate, gcp_picker_save_array
from scripts.mavlink import process_mavlink_log_for_webapp
from scripts.bin_to_images.bin_to_images import extract_binary, extraction_worker
from scripts.plot_marking.plot_marking import plot_marking_bp

# stitch pipeline
import sys
AGROWSTITCH_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../AgRowStitch"))
print(AGROWSTITCH_PATH)
sys.path.append(AGROWSTITCH_PATH)
from scripts.stitch_utils import (
    run_stitch_all_plots,
    monitor_stitch_updates_multi_plot,
    create_combined_mosaic_separate
)
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from PIL import ImageFile
from scripts.directory_index import DirectoryIndex, DirectoryIndexDict

# Paths to scripts
TRAIN_MODEL = os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts/deep_learning/model_training/train.py'))
LOCATE_PLANTS = os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts/deep_learning/trait_extraction/locate.py'))
EXTRACT_TRAITS = os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts/deep_learning/trait_extraction/extract.py'))

# Define the Flask application for serving files
file_app = Flask(__name__)
file_app.register_blueprint(plot_marking_bp)

latest_data = {'epoch': 0, 'map': 0, 'locate': 0, 'extract': 0, 'ortho': 0, 'drone_extract': 0}
training_stopped_event = threading.Event()
extraction_processes = {}
extraction_status = "not_started"  # Possible values: not_started, in_progress, done, failed
extraction_error_message = None  # Stores detailed error message if extraction fails
odm_method = None
stitch_thread=None
stitch_stop_event = threading.Event()

def complete_stitch_workflow(msgs_synced_path, image_path, config_path, custom_options, 
                            save_path, image_calibration, stitch_stop_event, progress_callback, monitoring_stop_event=None):
    """
    Complete workflow function that handles both stitching and mosaic creation
    """
    try:
        # Run the main stitching process
        print("=== STARTING MAIN STITCHING PROCESS ===")
        stitch_results = run_stitch_all_plots(
            msgs_synced_path, image_path, config_path, custom_options, 
            save_path, image_calibration, stitch_stop_event, progress_callback
        )
        
        # Check if stitching was successful and we have results
        if stitch_results and stitch_results.get('processed_plots'):
            print("=== MAIN STITCHING COMPLETED, STARTING MOSAIC CREATION ===")
            
            # Create the combined mosaic separately to avoid multiprocessing issues
            mosaic_success = create_combined_mosaic_separate(
                stitch_results['versioned_output_path'],
                stitch_results['processed_plots'],
                progress_callback,
                monitoring_stop_event  # Pass the monitoring stop event to cancel monitoring
            )
            
            if mosaic_success:
                print("=== COMPLETE WORKFLOW FINISHED SUCCESSFULLY ===")
            else:
                print("=== WORKFLOW COMPLETED WITH MOSAIC WARNINGS ===")
        else:
            print("=== WORKFLOW COMPLETED WITH ERRORS ===")
                
    except Exception as e:
        print(f"ERROR in complete_stitch_workflow: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
    finally:
        print("=== WORKFLOW FINALLY BLOCK - ENSURING COMPLETION ===")
        
        # Set final 100% progress - monitoring thread is already stopped by mosaic creation
        if progress_callback:
            print("Setting final progress to 100%...")
            progress_callback(100)
            print("Final progress set to 100% - workflow complete!")
        
        print("=== WORKFLOW FULLY COMPLETED ===")

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
    # global data_root_dir
    return send_from_directory(data_root_dir, filename)

# endpoint to serve image in memory
@file_app.route('/images/<path:filename>')
def serve_image(filename):
    global image_dict
    return image_dict[filename]

# endpoint to serve PNG files directly
@file_app.route('/get_png_file', methods=['POST'])
def get_png_file():
    data = request.json
    png_path = data['filePath']
    
    # Construct the full file path
    png_full_path = os.path.join(data_root_dir, png_path)
    
    if not os.path.exists(png_full_path):
        return jsonify({'error': 'PNG file not found'}), 404
    
    try:
        # Send the PNG file directly
        return send_file(png_full_path, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@file_app.route('/fetch_data_root_dir', methods=['GET'])
def fetch_data_root_dir():
    # global data_root_dir
    return data_root_dir

# endpoint to list directories
@file_app.route('/list_dirs/<path:dir_path>', methods=['GET'])
def list_dirs(dir_path):
    """Fast directory listing using index"""
    global data_root_dir, dir_db
    
    full_path = os.path.join(data_root_dir, dir_path)

    # Try index first
    dirs = dir_db.get_children(full_path, directories_only=True, wait_if_needed=True)

    return jsonify(dirs), 200

@file_app.get("/list_dirs_nested")
async def list_dirs_nested():
    global data_root_dir, dir_db
    
    base_dir = Path(data_root_dir) / 'Raw'
    
    try:
        print(f"Getting nested structure for Raw directory using DirectoryIndex: {base_dir}")
        
        # Try database-first approach
        nested_structure = await process_directories_in_parallel_from_db(dir_db, base_dir, max_depth=9)
        
        return jsonify(nested_structure), 200
        
    except Exception as e:
        print(f"Error getting nested structure from database: {e}")
        print("Falling back to original filesystem method...")
        
        # Fallback to original method if database approach fails
        try:
            nested_structure = await process_directories_in_parallel(base_dir, max_depth=9)
            return jsonify(nested_structure), 200
        except Exception as fallback_error:
            print(f"Error in fallback method: {fallback_error}")
            return jsonify({'error': 'Failed to get directory structure'}), 500

@file_app.get("/list_dirs_nested_processed")
async def list_dirs_nested_processed():
    global data_root_dir, dir_db

    base_dir = Path(data_root_dir) / 'Processed'
    
    try:
        print(f"Getting nested structure for Processed directory using DirectoryIndex: {base_dir}")
        
        # Try database-first approach
        nested_structure = await process_directories_in_parallel_from_db(base_dir, max_depth=9)
        
        return jsonify(nested_structure), 200
        
    except Exception as e:
        print(f"Error getting nested structure from database: {e}")
        print("Falling back to original filesystem method...")
        
        # Fallback to original method if database approach fails
        try:
            nested_structure = await process_directories_in_parallel(base_dir, max_depth=9)
            return jsonify(nested_structure), 200
        except Exception as fallback_error:
            print(f"Error in fallback method: {fallback_error}")
            return jsonify({'error': 'Failed to get directory structure'}), 500

# endpoint to list files
@file_app.route('/list_files/<path:dir_path>', methods=['GET'])
def list_files(dir_path):
    """Fast file listing using directory index"""
    global data_root_dir, dir_db
    
    full_path = os.path.join(data_root_dir, dir_path)
    
    # Try to get files from directory index (both files and directories, then filter)
    try:
        all_items = dir_db.get_children(full_path, directories_only=False, wait_if_needed=True)
        
        # Filter to get only files (not directories)
        if all_items and isinstance(all_items[0], dict):
            # If items have type information
            files = [item['name'] for item in all_items if not item.get('is_directory', True)]
        else:
            # If no items returned from index, use fallback
            files = []
    except Exception as e:
        print(f"Error using directory index for files: {e}")
        files = []
    
    # Enhanced fallback with error handling
    if not files and os.path.exists(full_path):
        try:
            # Direct filesystem read as fallback
            all_entries = os.listdir(full_path)
            files = []
            for entry in all_entries:
                if not entry.startswith('.'):  # Skip hidden files
                    entry_path = os.path.join(full_path, entry)
                    if os.path.isfile(entry_path):  # Only include files
                        files.append(entry)
            
            files.sort()
            
            # Queue for background processing to update the database
            if hasattr(dir_db, 'refresh_queue'):
                dir_db.refresh_queue.put(full_path)
            
        except PermissionError:
            print(f"Permission denied accessing: {full_path}")
            return jsonify({'error': 'Permission denied'}), 403
        except Exception as e:
            print(f"Error reading directory {full_path}: {e}")
            return jsonify({'error': 'Directory read failed'}), 500
    
    if not os.path.exists(full_path):
        return jsonify({'message': 'Directory not found'}), 404
    
    return jsonify(files), 200

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
    # global data_root_dir

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
    # global data_root_dir

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
    # global data_root_dir
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
    
    # Sanitize the directory path to remove any hidden Unicode characters    
    # Normalize Unicode and remove control characters
    dir_path_clean = unicodedata.normalize('NFKD', dir_path)
    dir_path_clean = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', dir_path_clean)  # Remove control characters
    dir_path_clean = re.sub(r'[^\x20-\x7e]', '', dir_path_clean)  # Keep only ASCII printable characters
    dir_path_clean = dir_path_clean.strip()  # Remove leading/trailing whitespace
    
    if dir_path != dir_path_clean:
        print(f"Original dir_path: {repr(dir_path)}")
        print(f"Cleaned dir_path: {repr(dir_path_clean)}")
    
    full_dir_path = os.path.join(UPLOAD_BASE_DIR, dir_path_clean)
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
        thread.daemon = True  # Set as daemon thread to terminate with main thread
        thread.start()

    if data_type.lower() == "platformlogs":
        msgs_synced_file = os.path.join(os.path.dirname(full_dir_path), "drone_msgs.csv")
        if os.path.isfile(msgs_synced_file):
            existing_df = pd.read_csv(msgs_synced_file)
            existing_paths = set(existing_df['timestamp'].values)
        else:
            existing_paths = set()
        thread = threading.Thread(
            target=process_mavlink_log_for_webapp, 
            args=(uploaded_file_paths, data_type, msgs_synced_file, existing_df, existing_paths)
        )
        thread.daemon = True  # Set as daemon thread to terminate with main thread
        thread.start()

    # Update directory database after upload completion
    if uploaded_file_paths and dir_db is not None:
        try:
            # Refresh the directory in the database
            dir_db.force_refresh(full_dir_path)
            print(f"Updated directory database for: {full_dir_path}")
        except Exception as e:
            print(f"Error updating directory database: {e}")

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
    if not os.path.exists(report_path):
        report_path = f"{UPLOAD_BASE_DIR}/{data['year']}/{data['experiment']}/{data['location']}/{data['population']}/{data['date']}/Amiga/RGB/report.txt"
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


@file_app.route('/get_binary_status', methods=['GET'])
def get_binary_status():
    print(f"Extraction status: {extraction_status}")
    response = {'status': extraction_status}
    if extraction_status == "failed" and extraction_error_message:
        response['error_message'] = extraction_error_message
    return jsonify(response), 200

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
    p = Process(target=extraction_worker, args=(file_paths, output_path), daemon=True)
    p.start()
    extraction_processes[dir_path] = p

    return jsonify({'status': 'started'}), 200

@file_app.route('/get_binary_progress', methods=['POST'])
def get_binary_progress():
    dir_path = request.json['localDirPath']
    dir_path = os.path.join(UPLOAD_BASE_DIR, dir_path)
    try:
        for root, dirs, files in os.walk(dir_path):
            if 'progress.txt' in files:
                progress_file_path = os.path.join(root, 'progress.txt')
                with open(progress_file_path, 'r') as file:
                    content = file.read().strip()
                try:
                    # allow float fractional progress
                    progress = float(content)
                except ValueError:
                    progress = 0.0
                print(f'Binary extraction progress (raw): {progress}')
                return jsonify({'progress': progress}), 200
        return jsonify({'progress': 0, 'error': 'progress.txt not found'}), 404
    except Exception as e:
        return jsonify({'progress': 0, 'error': str(e)}), 500

@file_app.route('/upload_chunk', methods=['POST'])
def upload_chunk():
    chunk = request.files['fileChunk']
    chunk_index = int(request.form['chunkIndex'])
    total_chunks = int(request.form['totalChunks'])
    file_name = secure_filename(request.form['fileIdentifier'])
    dir_path = request.form['dirPath']
    
    # Sanitize the directory path to remove any hidden Unicode characters
    # Normalize Unicode and remove control characters
    dir_path_clean = unicodedata.normalize('NFKD', dir_path)
    dir_path_clean = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', dir_path_clean)  # Remove control characters
    dir_path_clean = re.sub(r'[^\x20-\x7e]', '', dir_path_clean)  # Keep only ASCII printable characters
    dir_path_clean = dir_path_clean.strip()  # Remove leading/trailing whitespace
    
    print(f"Chunk upload - Original dir_path: {repr(dir_path)}")
    print(f"Chunk upload - Cleaned dir_path: {repr(dir_path_clean)}")
    
    full_dir_path = os.path.join(UPLOAD_BASE_DIR, dir_path_clean)
    cache_dir_path = os.path.join(full_dir_path, 'cache')
    os.makedirs(full_dir_path, exist_ok=True)
    os.makedirs(cache_dir_path, exist_ok=True)

    chunk_save_path = os.path.join(cache_dir_path, f"{file_name}.part{chunk_index}")
    chunk.save(chunk_save_path)

    print(f"Chunk {chunk_index} of {total_chunks} received")

    # Only reassemble if this is the last chunk
    # (client uploads in any order, so check all parts)
    all_parts = [os.path.exists(os.path.join(cache_dir_path, f"{file_name}.part{i}")) for i in range(total_chunks)]
    if all(all_parts):
        print("Reassembling file...")
        assembled_path = os.path.join(full_dir_path, file_name)
        try:
            with open(assembled_path + ".tmp", 'wb') as full_file:
                for i in range(total_chunks):
                    part_path = os.path.join(cache_dir_path, f"{file_name}.part{i}")
                    with open(part_path, 'rb') as part_file:
                        shutil.copyfileobj(part_file, full_file)
            os.replace(assembled_path + ".tmp", assembled_path)  # atomic move
            print("Finished reassembling file...")
            
            # Optionally, cleanup parts here
            for i in range(total_chunks):
                os.remove(os.path.join(cache_dir_path, f"{file_name}.part{i}"))
            
            # Update directory database after successful file assembly
            if dir_db is not None:
                try:
                    dir_db.force_refresh(full_dir_path)
                    print(f"Updated directory database for: {full_dir_path}")
                except Exception as e:
                    print(f"Error updating directory database: {e}")
                    
            return "File reassembled and saved successfully", 200
        except Exception as e:
            print(f"Error during reassembly: {e}")
            return f"Error during reassembly: {e}", 500
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
    # global data_root_dir
    try:
        image_folder = os.path.join(data_root_dir, 'Raw', request.json['year'], 
                                    request.json['experiment'], request.json['location'], 
                                    request.json['population'], request.json['date'], 
                                    request.json['platform'], request.json['sensor'], 'Images')
        
        # if folder 'top' is in image_folder, add it to the path
        if os.path.isdir(os.path.join(image_folder, 'top')):
            print("Found 'top' folder in image_folder, adding it to the path")
            image_folder = os.path.join(image_folder, 'top')
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
    # global data_root_dir
    try:
        image_folder = os.path.join(data_root_dir, 'Raw', request.json['year'], 
                                    request.json['experiment'], request.json['location'], 
                                    request.json['population'], request.json['date'], 
                                    request.json['platform'], request.json['sensor'], 'Images')
        # if folder 'top' is in image_folder, add it to the path
        if os.path.isdir(os.path.join(image_folder, 'top')):
            print("Found 'top' folder in image_folder, adding it to the path")
            image_folder = os.path.join(image_folder, 'top')
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
    global processed_image_folder
    global now_drone_processing
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
            
            # Check if pointX and pointY are not null
            if item['pointX'] is not None and item['pointY'] is not None:
                # Add or update the point
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
            else:
                # Remove the point if pointX or pointY is null
                if image_name in existing_data:
                    del existing_data[image_name]
                    if debug:
                        print(f"Removed point for image: {image_name}")

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
    platform = request.json['platform']
    orthomosaic_version = request.json.get('orthomosaic_version')  # Optional parameter for specific version

    prefix = data_root_dir+'/Processed'
    
    # If orthomosaic_version is provided, look in the version-specific directory
    if orthomosaic_version:
        traitpth = os.path.join(prefix, year, experiment, location, population, date, platform, sensor, orthomosaic_version,
                              f"{date}-{sensor}-{orthomosaic_version}-Traits-WGS84.geojson")

    if not os.path.isfile(traitpth):
        return jsonify({'message': []}), 404
    else:
        gdf = gpd.read_file(traitpth)
        traits = list(gdf.columns)
        extraneous_columns = ['Tier','Bed','Plot','Label','Group','geometry']
        traits = [x for x in traits if x not in extraneous_columns]
        print(traits, flush=True)
        return jsonify(traits), 200

@file_app.route('/get_orthomosaic_versions', methods=['POST'])
def get_orthomosaic_versions():
    """
    Get available orthomosaic versions for a specific dataset
    Returns both aerial/drone traits (sensor level) and roboflow inference traits (version level)
    """
    try:
        data = request.json
        year = data.get('year')
        experiment = data.get('experiment')
        location = data.get('location')
        population = data.get('population')
        date = data.get('date')
        platform = data.get('platform')
        sensor = data.get('sensor')
        
        if not all([year, experiment, location, population, date, platform, sensor]):
            return jsonify({"error": "Missing required parameters"}), 400
            
        versions = []
        
        # Check for AgRowStitch versions
        processed_path = os.path.join(data_root_dir, 'Processed', year, experiment, location, population, date, platform, sensor)
        if os.path.exists(processed_path):
            for item in os.listdir(processed_path):
                item_path = os.path.join(processed_path, item)
                if os.path.isdir(item_path) and item.startswith('AgRowStitch_v'):
                    versions.append({'type': 'AgRowStitch', 'AGR_version': item})

        # Check for ODM orthomosaics for splitting if not done in get plot images
        # if os.path.exists(processed_path):
        #     for file in os.listdir(processed_path):
        #         if file.endswith('-RGB.tif'):
        #             versions.append('ODM_Direct')
        #             break
        
        # Check for plot images from split_orthomosaics
        intermediate_path = os.path.join(data_root_dir, 'Intermediate', year, experiment, location, population, 'plot_images', date)
        if os.path.exists(intermediate_path):
            plot_files = [f for f in os.listdir(intermediate_path) if f.startswith('plot_') and f.endswith('.png')]
            if plot_files:
                versions.append({'type': 'Plot_Images', 'AGR_version': 'Plot_Images'})

        prefix = data_root_dir + '/Processed'
        sensor_dir = os.path.join(prefix, year, experiment, location, population, date, platform, sensor)
        
        # versions = []
        
        if os.path.exists(sensor_dir):
            # Check for aerial/drone traits at sensor level
            aerial_trait_file = f"{date}-{platform}-{sensor}-Traits-WGS84.geojson"
            aerial_trait_path = os.path.join(sensor_dir, aerial_trait_file)
            
            if os.path.exists(aerial_trait_path):
                # Convert to relative path for Flask file server
                relative_path = os.path.relpath(aerial_trait_path, data_root_dir)
                versions.append({
                    'type': 'aerial',
                    'version': 'main',
                    'versionName': f'{platform}',
                    'versionType': 'aerial',
                    'path': f'/files/{relative_path.replace(os.sep, "/")}'
                })
            
            # Check for roboflow inference traits in subdirectories
            for item in os.listdir(sensor_dir):
                item_path = os.path.join(sensor_dir, item)
                if os.path.isdir(item_path):
                    # Look for traits file with version-specific naming
                    trait_file = f"{date}-{platform}-{sensor}-{item}-Traits-WGS84.geojson"
                    trait_path = os.path.join(item_path, trait_file)
                    
                    if os.path.exists(trait_path):
                        # Convert to relative path for Flask file server
                        relative_path = os.path.relpath(trait_path, data_root_dir)
                        versions.append({
                            'type': 'roboflow',
                            'version': item,
                            'versionName': f'{item}',
                            'versionType': 'roboflow',
                            'path': f'/files/{relative_path.replace(os.sep, "/")}'
                        })
        print(f"Available orthomosaic versions: {versions}")
        return jsonify(versions), 200
        
    except Exception as e:
        print(f"Error getting orthomosaic versions: {e}")
        return jsonify({'error': str(e)}), 500

@file_app.route('/get_agrowstitch_versions', methods=['POST'])
def get_agrowstitch_versions():
    """
    Alias for get_orthomosaic_versions for backward compatibility
    """
    return get_orthomosaic_versions()
    
def select_middle(df):
    middle_index = len(df) // 2  # Find the middle index
    return df.iloc[[middle_index]]  # Use iloc to select the middle row

def filter_images(geojson_features, year, experiment, location, population, date, platform, sensor, middle_image=False):

    # global data_root_dir

    # Construct the CSV path from the state variables
    csv_path = os.path.join(data_root_dir, 'Raw', year, experiment, location, 
                            population, date, 'rover', 'RGB', 'Metadata', 'msgs_synced.csv')
    df = pd.read_csv(csv_path)

    # Convert DataFrame to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))

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
    sensor_col = '/top/' + sensor.lower()
    filtered_images = filtered_gdf[sensor_col+"_file"].tolist()
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
    
@file_app.route('/get_odm_logs', methods=['POST'])
def get_odm_logs():
    
    # get data
    data = request.json
    method = data.get('method')
    ortho_data = data.get('orthoData')
    
    print("Method for logs:", method)  # Debug statement
    
    if method == 'STITCH':
        final_mosaics_path = os.path.join(
            data_root_dir, "Raw", ortho_data['year'],
            ortho_data['experiment'], ortho_data['location'], ortho_data['population'],
            ortho_data['date'], ortho_data['platform'], ortho_data['sensor'], 
            "Images", "final_mosaics"
        )
        
        if not os.path.exists(final_mosaics_path):
            return jsonify({"error": "Final mosaics directory not found"}), 404
        
        # Find all plot log files
        log_files = []
        for file in os.listdir(final_mosaics_path):
            if file.startswith("temp_plot_") and file.endswith(".log"):
                log_files.append(os.path.join(final_mosaics_path, file))
        
        if not log_files:
            return jsonify({"error": "No plot log files found"}), 404
        
        # Sort log files by plot ID for consistent order
        log_files.sort()
        
        # Combine logs from all plots
        combined_logs = []
        for log_file in log_files:
            plot_name = os.path.basename(log_file).replace(".log", "")
            combined_logs.append(f"\n=== {plot_name.upper()} LOG ===\n")
            
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    combined_logs.append(content)
                    if not content.endswith('\n'):
                        combined_logs.append('\n')
            except Exception as e:
                combined_logs.append(f"Error reading {log_file}: {str(e)}\n")
        
        return jsonify({"log_content": ''.join(combined_logs)}), 200
    else:
        # Original ODM logs logic
        logs_path = os.path.join(data_root_dir, 'temp', 'project', 'code', 'logs.txt')
        print("Logs Path:", logs_path)  # Debug statement
        if os.path.exists(logs_path):
            with open(logs_path, 'r') as log_file:
                lines = log_file.readlines()
                # latest_logs = lines[-20:]  # Get the last 20 lines
            return jsonify({"log_content": ''.join(lines)}), 200
        else:
            print("Logs not found at path:", logs_path)  # Debug statement
            return jsonify({"error": "Logs not found"}), 404

@file_app.route('/run_stitch', methods=['POST'])
def run_stitch_endpoint():
    data = request.json
    location = data.get('location')
    population = data.get('population')
    date = data.get('date')
    year = data.get('year')
    experiment = data.get('experiment')
    platform = data.get('platform')
    sensor = data.get('sensor')
    # temp_dir = data.get('temp_dir')
    # reconstruction_quality = data.get('reconstruction_quality')
    custom_options = data.get('custom_options')
    
    global stitch_thread, stitch_stop_event, odm_method
    global stitched_path, temp_output
    odm_method = 'stitch'
    stitch_stop_event.clear()
    
    try:
        image_path = os.path.join(
            data_root_dir, "Raw", year,
            experiment, location, population,
            date, platform, sensor, "Images", "top"
        )
        msgs_synced_path = os.path.join(
            data_root_dir, "Raw", year,
            experiment, location, population,
            date, platform, sensor, "Metadata", "msgs_synced.csv"
        )
        save_path = os.path.join(
            data_root_dir, "Processed", year,
            experiment, location, population,
            date, platform, sensor
        )
        image_calibration = os.path.join(
            data_root_dir, "Raw", year,
            experiment, location, population,
            date, platform, sensor, "Metadata", "top_calibration.json"
        )
        config_path = f"{AGROWSTITCH_PATH}/panorama_maker/config.yaml"
        stitched_path = os.path.join(os.path.dirname(image_path), "final_mosaics")
        temp_output = os.path.join(os.path.dirname(image_path), "top_output")
        
        # Check if image_path exists
        if not os.path.exists(image_path):
            return jsonify({
                "status": "error", 
                "message": f"Error: Image path not found at {image_path}\n\nThe selected images may not be compatible with AgRowStitch yet."
            }), 404
        
        # remove stitched_path and temp_output if it exists
        if os.path.exists(stitched_path):
            shutil.rmtree(stitched_path)
            print(f"Removed existing stitched path: {stitched_path}")
        if os.path.exists(temp_output):
            shutil.rmtree(temp_output)
            print(f"Removed existing temp output path: {temp_output}")
        
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        
        # Check if msgs_synced.csv exists
        if not os.path.exists(msgs_synced_path):
            return jsonify({"status": "error", "message": f"msgs_synced.csv not found at {msgs_synced_path}"}), 404
        
        # Load and validate plot indices
        import pandas as pd
        msgs_df = pd.read_csv(msgs_synced_path)
        
        # Check if plot_index column exists
        if 'plot_index' not in msgs_df.columns:
            return jsonify({
                "status": "error", 
                "message": "Plot index column not found in msgs_synced.csv. Please perform plot marking in File Management first."
            }), 400
        
        # Check if any plot indices are defined (excluding plot 0 which is usually background/unassigned)
        unique_plots = [pid for pid in msgs_df['plot_index'].unique() if pid != 0 and not pd.isna(pid)]
        
        if len(unique_plots) == 0:
            return jsonify({
                "status": "error", 
                "message": "No plot indices are defined in the dataset. Please perform plot marking in File Management to assign images to plots before running AgRowStitch."
            }), 400
        
        print(f"Found {len(unique_plots)} unique plots for stitching: {unique_plots}")
        
        # Create a monitoring stop event to coordinate threads
        monitoring_stop_event = threading.Event()
        
        # Start multi-plot log monitoring thread
        final_mosaics_dir = os.path.join(os.path.dirname(image_path), "final_mosaics")
        
        def progress_callback(progress):
            latest_data['ortho'] = progress
        
        thread_prog = threading.Thread(
            target=monitor_stitch_updates_multi_plot, 
            args=(final_mosaics_dir, unique_plots, progress_callback, monitoring_stop_event), 
            daemon=True
        )
        thread_prog.start()
        
        # Start the stitching process for all plots in background
        stitch_thread = threading.Thread(
            target=complete_stitch_workflow,
            args=(msgs_synced_path, image_path, config_path, custom_options, 
                  save_path, image_calibration, stitch_stop_event, progress_callback, monitoring_stop_event),
            daemon=True
        )
        stitch_thread.start()

        return jsonify({"status": "started", "message": "Stitching process started for all plots"}), 202

    except Exception as e:
        print(f"Error running AgRowStitch: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

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
        thread_prog = threading.Thread(target=monitor_log_updates, args=(latest_data, logs_path, progress_file), daemon=True)
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
    # global data_root_dir, stitch_thread, stitch_stop_event, odm_method, stitched_path, temp_output
    try:
        if odm_method == 'stitch' and stitch_thread is not None:
            print("Stopping stitching thread...")
            stitch_stop_event.set()
            stitch_thread.join(timeout=10)  # Wait up to 10s for clean exit
            print("Stitching stopped.")
            
            # remove stitched_path and temp_output if it exists
            if os.path.exists(stitched_path):
                shutil.rmtree(stitched_path)
                print(f"Removed existing stitched path: {stitched_path}")
            if os.path.exists(temp_output):
                shutil.rmtree(temp_output)
                print(f"Removed existing temp output path: {temp_output}")
        else:
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
    # global data_root_dir
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

@file_app.route('/download_plot_ortho', methods=['POST'])
def download_plot_ortho():
    data = request.get_json()
    try:
        # Build the path to the plot images directory
        plot_dir = os.path.join(
            data_root_dir,
            'Processed',
            data['year'],
            data['experiment'],
            data['location'],
            data['population'],
            data['date'],
            data['platform'],
            data['sensor'],
            data['agrowstitchDir']
        )
        
        if not os.path.exists(plot_dir):
            return jsonify({'error': 'Plot directory not found'}), 404
        
        # Find all plot PNG files
        plot_files = [f for f in os.listdir(plot_dir) 
                     if f.startswith('full_res_mosaic_temp_plot_') and f.endswith('.png')]
        
        if not plot_files:
            return jsonify({'error': 'No plot files found'}), 404
        
        # Get plot borders data for custom naming
        plot_borders_path = os.path.join(
            data_root_dir, "Raw", data['year'], data['experiment'], 
            data['location'], data['population'], "plot_borders.csv"
        )
        
        plot_data = {}
        if os.path.exists(plot_borders_path):
            try:
                borders_df = pd.read_csv(plot_borders_path)
                for _, row in borders_df.iterrows():
                    plot_index = row.get('plot_index')
                    plot_label = row.get('Plot')
                    accession = row.get('Accession')
                    
                    if not pd.isna(plot_index):
                        plot_data[int(plot_index)] = {
                            'plot_label': plot_label if not pd.isna(plot_label) else None,
                            'accession': accession if not pd.isna(accession) else None
                        }
            except Exception as e:
                print(f"Error reading plot borders for zip download: {e}")
        
        # Create a temporary zip file
        import tempfile
        import zipfile
        
        temp_dir = tempfile.mkdtemp()
        zip_filename = f"{data['date']}-{data['platform']}-{data['sensor']}-plots-with-metadata.zip"
        zip_path = os.path.join(temp_dir, zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for plot_file in sorted(plot_files):
                file_path = os.path.join(plot_dir, plot_file)
                
                # Extract plot index and create custom filename
                plot_match = re.search(r'temp_plot_(\d+)', plot_file)
                if plot_match:
                    plot_index = int(plot_match.group(1))
                    extension = os.path.splitext(plot_file)[1]
                    
                    # Get plot metadata
                    metadata = plot_data.get(plot_index, {})
                    plot_label = metadata.get('plot_label')
                    accession = metadata.get('accession')
                    
                    # Create custom filename using the requested format
                    if plot_label and accession:
                        custom_filename = f"plot_{plot_label}_accession_{accession}{extension}"
                    elif plot_label:
                        custom_filename = f"plot_{plot_label}{extension}"
                    else:
                        custom_filename = f"plot_{plot_index}{extension}"
                    
                    # Add file to zip with custom name
                    zipf.write(file_path, custom_filename)
                    print(f"Adding to zip: {plot_file} -> {custom_filename}")
                else:
                    # Fallback to original filename if pattern doesn't match
                    zipf.write(file_path, plot_file)
        
        print(f"Sending zip file: {zip_path}")
        return send_file(
            zip_path,
            mimetype="application/zip",
            as_attachment=True,
            download_name=zip_filename
        )
    
    except Exception as e:
        print(f"An error occurred while downloading plot ortho: {str(e)}")
        return jsonify({'error': str(e)}), 500
        

@file_app.route('/delete_ortho', methods=['POST'])
def delete_ortho():
    # global data_root_dir
    data = request.json
    year = data.get('year')
    experiment = data.get('experiment')
    location = data.get('location')
    population = data.get('population')
    date = data.get('date')
    platform = data.get('platform')
    sensor = data.get('sensor')
    delete_type = data.get('deleteType', 'ortho')  # 'ortho' or 'agrowstitch'
    
    base_path = os.path.join(data_root_dir, 'Processed', year, experiment, location, population, date, platform, sensor)
    
    try:
        if delete_type == 'agrowstitch':
            # Delete specific AgRowStitch version directory
            agrowstitch_dir = data.get('agrowstitchDir')
            if agrowstitch_dir:
                agrowstitch_path = os.path.join(base_path, agrowstitch_dir)
                if os.path.exists(agrowstitch_path):
                    shutil.rmtree(agrowstitch_path)
                    print(f"Deleted AgRowStitch directory: {agrowstitch_path}")
                else:
                    print(f"AgRowStitch directory not found: {agrowstitch_path}")
            else:
                return jsonify({"error": "AgRowStitch directory name not provided"}), 400
        else:
            # Delete specific orthomosaic file
            file_name = data.get('fileName')
            if file_name:
                file_path = os.path.join(base_path, file_name)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                else:
                    print(f"File not found: {file_path}")
            else:
                # For backward compatibility, try to find and delete standard orthomosaic files
                # Look for common orthomosaic file patterns
                ortho_patterns = [
                    f"{date}-RGB-Pyramid.tif",
                    f"{date}-RGB.tif", 
                    f"{date}-RGB.png",
                    f"{date}-DEM-Pyramid.tif",
                    f"{date}-DEM.tif"
                ]
                
                deleted_files = []
                for pattern in ortho_patterns:
                    file_path = os.path.join(base_path, pattern)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        deleted_files.append(pattern)
                        print(f"Deleted file: {file_path}")
                
                if not deleted_files:
                    return jsonify({"error": "No orthomosaic files found to delete"}), 404
                
    except FileNotFoundError:
        print("Path not found during deletion")
        return jsonify({"error": "File or directory not found"}), 404
    except PermissionError:
        print("Permission denied during deletion")
        return jsonify({"error": "Permission denied"}), 403
    except Exception as e:
        print(f"An error occurred during deletion: {str(e)}")
        return jsonify({"error": f"Deletion failed: {str(e)}"}), 500
        
    return jsonify({"message": "Ortho deleted successfully"}), 200

@file_app.route('/associate_plots_with_boundaries', methods=['POST'])
def associate_plots_with_boundaries():
    """
    Associate AgRowStitch plots with boundary polygons and update plot_borders.csv with plot labels
    """
    import geopandas as gpd
    from shapely.geometry import Point
    import pandas as pd
    
    data = request.json
    year = data.get('year')
    experiment = data.get('experiment')
    location = data.get('location')
    population = data.get('population')
    date = data.get('date')
    platform = data.get('platform')
    sensor = data.get('sensor')
    agrowstitch_dir = data.get('agrowstitchDir')
    boundaries = data.get('boundaries')  # GeoJSON FeatureCollection
    
    if not all([year, experiment, location, population, date, platform, sensor, agrowstitch_dir, boundaries]):
        return jsonify({'error': 'Missing required parameters'}), 400
    
    try:
        # Paths
        msgs_synced_path = os.path.join(
            data_root_dir, "Raw", year, experiment, location, population,
            date, platform, sensor, "Metadata", "msgs_synced.csv"
        )
        plot_borders_path = os.path.join(
            data_root_dir, "Raw", year, experiment, location, population,
            "plot_borders.csv"
        )
        agrowstitch_path = os.path.join(
            data_root_dir, "Processed", year, experiment, location, population,
            date, platform, sensor, agrowstitch_dir
        )
        
        # Check if msgs_synced.csv exists
        if not os.path.exists(msgs_synced_path):
            return jsonify({'error': f'msgs_synced.csv not found at {msgs_synced_path}'}), 404
            
        # Check if plot_borders.csv exists
        if not os.path.exists(plot_borders_path):
            return jsonify({'error': f'plot_borders.csv not found at {plot_borders_path}'}), 404
            
        # Check if AgRowStitch directory exists
        if not os.path.exists(agrowstitch_path):
            return jsonify({'error': f'AgRowStitch directory not found at {agrowstitch_path}'}), 404
            
        # Load msgs_synced.csv
        msgs_df = pd.read_csv(msgs_synced_path)
        
        # Load plot_borders.csv
        borders_df = pd.read_csv(plot_borders_path)
        
        # Check if plot_index column exists
        if 'plot_index' not in msgs_df.columns:
            return jsonify({'error': 'plot_index column not found in msgs_synced.csv'}), 400
            
        if 'plot_index' not in borders_df.columns:
            return jsonify({'error': 'plot_index column not found in plot_borders.csv'}), 400
            
        # Get unique plot indices (excluding unassigned)
        plot_indices = [idx for idx in msgs_df['plot_index'].unique() if idx > 0 and not pd.isna(idx)]
        
        if len(plot_indices) == 0:
            return jsonify({'error': 'No plot indices found in msgs_synced.csv'}), 400
            
        # Create GeoDataFrame from boundaries
        boundaries_gdf = gpd.GeoDataFrame.from_features(boundaries['features'])
        boundaries_gdf.set_crs(epsg=4326, inplace=True)
        
        # Initialize Plot and Accession columns if they don't exist
        if 'Plot' not in borders_df.columns:
            borders_df['Plot'] = None
        if 'Accession' not in borders_df.columns:
            borders_df['Accession'] = None
            
        # Track associations to prevent duplicates
        plot_associations = {}
        
        # For each plot index, find its center point and associate with boundary
        for plot_idx in plot_indices:
            plot_data = msgs_df[msgs_df['plot_index'] == plot_idx]
            
            if plot_data.empty:
                continue
                
            # Calculate center point of plot based on GPS coordinates
            center_lat = plot_data['lat'].mean()
            center_lon = plot_data['lon'].mean()
            center_point = Point(center_lon, center_lat)
            
            # Find which boundary contains this point
            for _, boundary in boundaries_gdf.iterrows():
                if boundary.geometry.contains(center_point):
                    # Get plot and accession from boundary properties
                    boundary_plot = boundary.get('plot', boundary.get('Plot'))
                    boundary_accession = boundary.get('accession', boundary.get('Accession'))
                    
                    # Check if this boundary is already associated with another plot
                    boundary_key = f"{boundary_plot}_{boundary_accession}"
                    if boundary_key in plot_associations:
                        print(f"Warning: Boundary {boundary_key} already associated with plot {plot_associations[boundary_key]}")
                        continue
                        
                    # Update plot_borders.csv with Plot and Accession
                    borders_df.loc[borders_df['plot_index'] == plot_idx, 'Plot'] = boundary_plot
                    borders_df.loc[borders_df['plot_index'] == plot_idx, 'Accession'] = boundary_accession
                    
                    # Track association
                    plot_associations[boundary_key] = int(plot_idx)
                    
                    print(f"Associated plot index {plot_idx} with boundary {boundary_key} -> Plot: {boundary_plot}, Accession: {boundary_accession}")
                    break
                    
        # Save updated plot_borders.csv
        borders_df.to_csv(plot_borders_path, index=False)
        
        # Return association summary
        return jsonify({
            'message': 'Plot associations completed successfully',
            'associations': int(len(plot_associations)),
            'plot_associations': plot_associations
        }), 200
        
    except Exception as e:
        print(f"Error in plot association: {e}")
        return jsonify({'error': str(e)}), 500

@file_app.route('/get_agrowstitch_plot_associations', methods=['POST'])
def get_agrowstitch_plot_associations():
    """
    Get current plot associations for AgRowStitch plots from plot_borders.csv
    """
    data = request.json
    year = data.get('year')
    experiment = data.get('experiment')
    location = data.get('location')
    population = data.get('population')
    date = data.get('date')
    platform = data.get('platform')
    sensor = data.get('sensor')
    
    if not all([year, experiment, location, population, date, platform, sensor]):
        return jsonify({'error': 'Missing required parameters'}), 400
        
    try:
        # Path to msgs_synced.csv (for plot indices)
        msgs_synced_path = os.path.join(
            data_root_dir, "Raw", year, experiment, location, population,
            date, platform, sensor, "Metadata", "msgs_synced.csv"
        )
        
        # Path to plot_borders.csv (for plot labels)
        plot_borders_path = os.path.join(
            data_root_dir, "Raw", year, experiment, location, population,
            "plot_borders.csv"
        )
        
        if not os.path.exists(msgs_synced_path):
            return jsonify({'error': 'msgs_synced.csv not found'}), 404
            
        if not os.path.exists(plot_borders_path):
            return jsonify({'error': 'plot_borders.csv not found'}), 404
            
        # Load both files
        msgs_df = pd.read_csv(msgs_synced_path)
        borders_df = pd.read_csv(plot_borders_path)
        
        # Check required columns
        if 'plot_index' not in msgs_df.columns:
            return jsonify({'error': 'plot_index column not found in msgs_synced.csv'}), 400
            
        if 'plot_index' not in borders_df.columns:
            return jsonify({'error': 'plot_index column not found in plot_borders.csv'}), 400
            
        # Get plot associations from plot_borders.csv
        associations = {}
        for _, row in borders_df.iterrows():
            plot_idx = row['plot_index']
            if plot_idx > 0 and (pd.notna(row.get('Plot')) or pd.notna(row.get('Accession'))):
                # Get center coordinates from msgs_synced.csv
                plot_data = msgs_df[msgs_df['plot_index'] == plot_idx]
                if not plot_data.empty:
                    center_lat = plot_data['lat'].mean()
                    center_lon = plot_data['lon'].mean()
                    
                    # Create plot label
                    plot_value = row.get('Plot')
                    accession_value = row.get('Accession')
                    
                    plot_label = f"Plot_{plot_value if pd.notna(plot_value) else 'Unknown'}"
                    if pd.notna(accession_value):
                        plot_label += f"_Acc_{accession_value}"
                    
                    associations[str(int(plot_idx))] = {
                        'plot_label': plot_label,
                        'center_lat': float(center_lat) if pd.notna(center_lat) else None,
                        'center_lon': float(center_lon) if pd.notna(center_lon) else None
                    }
                    
        return jsonify({
            'associations': associations,
            'total_plots': int(len([idx for idx in msgs_df['plot_index'].unique() if idx > 0 and not pd.isna(idx)]))
        }), 200
        
    except Exception as e:
        print(f"Error getting plot associations: {e}")
        return jsonify({'error': str(e)}), 500

@file_app.route('/get_plot_borders_data', methods=['POST'])
def get_plot_borders_data():
    """
    Get plot borders data with plot labels and accessions
    """
    data = request.json
    year = data.get('year')
    experiment = data.get('experiment') 
    location = data.get('location')
    population = data.get('population')
    
    if not all([year, experiment, location, population]):
        return jsonify({'error': 'Missing required parameters'}), 400
        
    try:
        # Path to plot_borders.csv
        plot_borders_path = os.path.join(
            data_root_dir, "Raw", year, experiment, location, population,
            "plot_borders.csv"
        )
        
        if not os.path.exists(plot_borders_path):
            return jsonify({'error': 'plot_borders.csv not found'}), 404
            
        # Load plot_borders.csv
        borders_df = pd.read_csv(plot_borders_path)
        
        # Create a dictionary mapping plot_index to plot labels and accessions
        plot_data = {}
        for _, row in borders_df.iterrows():
            plot_idx = row.get('plot_index')
            if pd.notna(plot_idx) and plot_idx > 0:
                plot_data[int(plot_idx)] = {
                    'plot': row.get('Plot') if pd.notna(row.get('Plot')) else None,
                    'accession': row.get('Accession') if pd.notna(row.get('Accession')) else None
                }
                
        return jsonify({'plot_data': plot_data}), 200
        
    except Exception as e:
        print(f"Error getting plot borders data: {e}")
        return jsonify({'error': str(e)}), 500

@file_app.route('/download_single_plot', methods=['POST'])
def download_single_plot():
    """
    Download a single plot image with plot label and accession in filename
    """
    data = request.get_json()
    try:
        year = data['year']
        experiment = data['experiment']
        location = data['location']
        population = data['population']
        date = data['date']
        platform = data['platform']
        sensor = data['sensor']
        agrowstitch_dir = data['agrowstitchDir']
        plot_filename = data['plotFilename']
        
        # Extract plot index from filename
        plot_match = re.search(r'temp_plot_(\d+)', plot_filename)
        if not plot_match:
            return jsonify({'error': 'Could not extract plot index from filename'}), 400
        
        plot_index = int(plot_match.group(1))
        
        # Get plot borders data
        plot_borders_path = os.path.join(
            data_root_dir, "Raw", year, experiment, location, population,
            "plot_borders.csv"
        )
        
        plot_label = None
        accession = None
        
        if os.path.exists(plot_borders_path):
            try:
                borders_df = pd.read_csv(plot_borders_path)
                plot_row = borders_df[borders_df['plot_index'] == plot_index]
                if not plot_row.empty:
                    plot_label = plot_row.iloc[0].get('Plot')
                    accession = plot_row.iloc[0].get('Accession')
                    if pd.isna(plot_label):
                        plot_label = None
                    if pd.isna(accession):
                        accession = None
            except Exception as e:
                print(f"Error reading plot borders: {e}")
        
        # Build original file path
        file_path = os.path.join(
            data_root_dir, 'Processed', year, experiment, location, population,
            date, platform, sensor, agrowstitch_dir, plot_filename
        )
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'Plot file not found'}), 404
        
        # Create new filename with plot label and accession in the requested format
        base_name = os.path.splitext(plot_filename)[0]
        extension = os.path.splitext(plot_filename)[1]
        
        # Use the format: plot_{plot}_accession_{accession}
        if plot_label and accession:
            new_filename = f"plot_{plot_label}_accession_{accession}{extension}"
        elif plot_label:
            new_filename = f"plot_{plot_label}{extension}"
        else:
            # Fallback to plot index if no plot label available
            new_filename = f"plot_{plot_index}{extension}"
        
        print(f"Download debug - Original: {plot_filename}, New format: {new_filename}")
        print(f"Plot data - plot_index: {plot_index}, plot_label: {plot_label}, accession: {accession}")

        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, new_filename)
        
        try:
            # Copy original file to temp location with new name
            shutil.copy2(file_path, temp_file_path)
            print(f"Created temporary file: {temp_file_path}")
            
            # Send the temporary file with the enhanced filename
            response = send_file(
                temp_file_path,
                as_attachment=True,
                download_name=new_filename,
                mimetype='image/png'
            )
            
            # Clean up temp file after sending (Flask handles this automatically)
            print(f"Sending file with custom format name: {new_filename}")
            return response
            
        except Exception as temp_error:
            # Clean up temp directory if error occurs
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise temp_error
        
    except Exception as e:
        print(f"Error downloading single plot: {str(e)}")
        return jsonify({'error': str(e)}), 500



@file_app.route('/get_data_options', methods=['GET'])
def get_data_options():
    """
    Get all available years from the data directory
    """
    try:
        years = []
        raw_dir = os.path.join(data_root_dir, 'Raw')
        
        if os.path.exists(raw_dir):
            years = [item for item in os.listdir(raw_dir) 
                    if os.path.isdir(os.path.join(raw_dir, item)) and item.isdigit()]
            years.sort()
        
        return jsonify({'years': years}), 200
        
    except Exception as e:
        print(f"Error getting data options: {e}")
        return jsonify({'error': str(e)}), 500

@file_app.route('/get_experiments', methods=['POST'])
def get_experiments():
    """
    Get available experiments for a given year
    """
    try:
        data = request.json
        year = data.get('year')
        
        if not year:
            return jsonify({'error': 'Year is required'}), 400
            
        experiments = []
        year_dir = os.path.join(data_root_dir, 'Raw', year)
        
        if os.path.exists(year_dir):
            experiments = [item for item in os.listdir(year_dir) 
                          if os.path.isdir(os.path.join(year_dir, item))]
            experiments.sort()
        
        return jsonify({'experiments': experiments}), 200
        
    except Exception as e:
        print(f"Error getting experiments: {e}")
        return jsonify({'error': str(e)}), 500

@file_app.route('/get_locations', methods=['POST'])
def get_locations():
    """
    Get available locations for a given year and experiment
    """
    try:
        data = request.json
        year = data.get('year')
        experiment = data.get('experiment')
        
        if not all([year, experiment]):
            return jsonify({'error': 'Year and experiment are required'}), 400
            
        locations = []
        exp_dir = os.path.join(data_root_dir, 'Raw', year, experiment)
        
        if os.path.exists(exp_dir):
            locations = [item for item in os.listdir(exp_dir) 
                        if os.path.isdir(os.path.join(exp_dir, item))]
            locations.sort()
        
        return jsonify({'locations': locations}), 200
        
    except Exception as e:
        print(f"Error getting locations: {e}")
        return jsonify({'error': str(e)}), 500

@file_app.route('/get_populations', methods=['POST'])
def get_populations():
    """
    Get available populations for a given year, experiment, and location
    """
    try:
        data = request.json
        year = data.get('year')
        experiment = data.get('experiment')
        location = data.get('location')
        
        if not all([year, experiment, location]):
            return jsonify({'error': 'Year, experiment, and location are required'}), 400
            
        populations = []
        loc_dir = os.path.join(data_root_dir, 'Raw', year, experiment, location)
        
        if os.path.exists(loc_dir):
            populations = [item for item in os.listdir(loc_dir) 
                          if os.path.isdir(os.path.join(loc_dir, item))]
            populations.sort()
        
        return jsonify({'populations': populations}), 200
        
    except Exception as e:
        print(f"Error getting populations: {e}")
        return jsonify({'error': str(e)}), 500

@file_app.route('/get_dates', methods=['POST'])
def get_dates():
    """
    Get available dates for a given year, experiment, location, and population
    """
    try:
        data = request.json
        year = data.get('year')
        experiment = data.get('experiment')
        location = data.get('location')
        population = data.get('population')
        
        if not all([year, experiment, location, population]):
            return jsonify({'error': 'Year, experiment, location, and population are required'}), 400
            
        dates = []
        # Check both Raw and Processed directories for dates
        raw_pop_dir = os.path.join(data_root_dir, 'Raw', year, experiment, location, population)
        processed_pop_dir = os.path.join(data_root_dir, 'Processed', year, experiment, location, population)
        
        date_set = set()
        
        # Check Raw directory
        if os.path.exists(raw_pop_dir):
            for item in os.listdir(raw_pop_dir):
                item_path = os.path.join(raw_pop_dir, item)
                if os.path.isdir(item_path) and item not in ['plot_borders.csv']:
                    date_set.add(item)
        
        # Check Processed directory
        if os.path.exists(processed_pop_dir):
            for item in os.listdir(processed_pop_dir):
                item_path = os.path.join(processed_pop_dir, item)
                if os.path.isdir(item_path):
                    date_set.add(item)
        
        dates = sorted(list(date_set))
        
        return jsonify({'dates': dates}), 200
        
    except Exception as e:
        print(f"Error getting dates: {e}")
        return jsonify({'error': str(e)}), 500

@file_app.route('/get_platforms', methods=['POST'])
def get_platforms():
    """
    Get available platforms for a given dataset
    """
    try:
        data = request.json
        year = data.get('year')
        experiment = data.get('experiment')
        location = data.get('location')
        population = data.get('population')
        date = data.get('date')
        
        if not all([year, experiment, location, population, date]):
            return jsonify({'error': 'All parameters are required'}), 400
            
        platforms = []
        # Check Processed directory for platforms
        date_dir = os.path.join(data_root_dir, 'Processed', year, experiment, location, population, date)
        
        if os.path.exists(date_dir):
            platforms = [item for item in os.listdir(date_dir) 
                        if os.path.isdir(os.path.join(date_dir, item))]
            platforms.sort()
        
        return jsonify({'platforms': platforms}), 200
        
    except Exception as e:
        print(f"Error getting platforms: {e}")
        return jsonify({'error': str(e)}), 500

@file_app.route('/get_sensors', methods=['POST'])
def get_sensors():
    """
    Get available sensors for a given dataset and platform
    """
    try:
        data = request.json
        year = data.get('year')
        experiment = data.get('experiment')
        location = data.get('location')
        population = data.get('population')
        date = data.get('date')
        platform = data.get('platform')
        
        if not all([year, experiment, location, population, date, platform]):
            return jsonify({'error': 'All parameters are required'}), 400
            
        sensors = []
        # Check Processed directory for sensors
        platform_dir = os.path.join(data_root_dir, 'Processed', year, experiment, location, population, date, platform)
        
        if os.path.exists(platform_dir):
            sensors = [item for item in os.listdir(platform_dir) 
                      if os.path.isdir(os.path.join(platform_dir, item))]
            sensors.sort()
        
        return jsonify({'sensors': sensors}), 200
        
    except Exception as e:
        print(f"Error getting sensors: {e}")
        return jsonify({'error': str(e)}), 500
               

        
### CVAT #### 
@file_app.route('/start_cvat', methods=['POST'])
def start_cvat():
    # global data_root_dir
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
    # global data_root_dir
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
    # global data_root_dir
    
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
    dir_path = request.form.get('dirPath')
    full_dir_path = os.path.join(data_root_dir, dir_path)
    os.makedirs(full_dir_path, exist_ok=True)

    uploaded_files = []
    for file in request.files.getlist("files"):
        filename = secure_filename(file.filename)
        file_path = os.path.join(full_dir_path, filename)
        print(f'Saving {file_path}...')
        file.save(file_path)
        uploaded_files.append(file_path)

    # Update directory database after upload completion
    if uploaded_files and dir_db is not None:
        try:
            dir_db.force_refresh(full_dir_path)
            print(f"Updated directory database for: {full_dir_path}")
        except Exception as e:
            print(f"Error updating directory database: {e}")

    return jsonify({'message': 'Files uploaded successfully'}), 200





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
    # global data_root_dir, latest_data, training_stopped_event, new_folder, train_labels, training_process
    
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



@file_app.route('/locate_plants', methods=['POST'])
def locate_plants():
    # global data_root_dir, save_locate, locate_process
    
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
    # global data_root_dir, save_extract, temp_extract, model_id, summary_date, locate_id, trait_extract, extract_process
    
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
        model_path = Path(data_root_dir)/'Intermediate'/year/experiment/location/population/'Training'/f'{platform}'/'RGB Plant Detection'/f'Plant-{id}'/'weights'/'last.pt' # TODO: DEBUG
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

@file_app.route('/split_orthomosaics', methods=['POST'])
def split_orthomosaics():
    """
    Split orthomosaics into individual plot images based on plot boundaries
    """
    try:
        data = request.get_json()
        year = data['year']
        experiment = data['experiment']
        location = data['location']
        population = data['population']
        date = data['date']
        boundaries = data['boundaries']
        
        # Construct paths
        base_path = os.path.join(data_root_dir, 'Processed', year, experiment, location, population, date)
        intermediate_path = os.path.join(data_root_dir, 'Intermediate', year, experiment, location, population)
        
        # Find orthomosaic
        orthomosaic_path = None
        platform_name = None
        sensor_name = None
        
        for platform in os.listdir(base_path):
            platform_path = os.path.join(base_path, platform)
            if not os.path.isdir(platform_path):
                continue
                
            for sensor in os.listdir(platform_path):
                sensor_path = os.path.join(platform_path, sensor)
                if not os.path.isdir(sensor_path):
                    continue
                    
                for file in os.listdir(sensor_path):
                    if file.endswith('-RGB.tif'):
                        orthomosaic_path = os.path.join(sensor_path, file)
                        platform_name = platform
                        sensor_name = sensor
                        break
                        
                if orthomosaic_path:
                    break
            if orthomosaic_path:
                break
        
        if not orthomosaic_path:
            return jsonify({"error": "No RGB orthomosaic found for the specified date"}), 404
        
        # Create output directory for plot images
        output_dir = os.path.join(intermediate_path, 'plot_images', date)
        os.makedirs(output_dir, exist_ok=True)
        
        plots_processed = 0
        
        # Process the orthomosaic
        try:
            import rasterio
            from rasterio.windows import from_bounds
            from rasterio.warp import transform_bounds
            import numpy as np
            from PIL import Image
            
            with rasterio.open(orthomosaic_path) as src:
                print(f"Opened orthomosaic: {orthomosaic_path}")
                print(f"CRS: {src.crs}, Shape: {src.shape}, Transform: {src.transform}")
                
                # Process each plot boundary
                for feature in boundaries['features']:
                    properties = feature['properties']
                    geometry = feature['geometry']
                    
                    # Get plot and accession info
                    plot = properties.get('plot', properties.get('Plot', 'unknown'))
                    accession = properties.get('accession', 'unknown')
                    
                    if plot == 'unknown' or accession == 'unknown':
                        print(f"Skipping feature - plot: {plot}, accession: {accession}")
                        continue
                        
                    # Validate geometry
                    if not geometry or geometry.get('type') != 'Polygon':
                        print(f"Invalid geometry for plot {plot}")
                        continue
                        
                    # Convert geometry coordinates to image coordinates
                    coords = geometry['coordinates'][0]  # Polygon exterior ring
                    
                    if len(coords) < 4:  # A polygon needs at least 4 points (including closing point)
                        print(f"Invalid polygon for plot {plot}: only {len(coords)} coordinates")
                        continue
                    
                    # Get bounding box from polygon
                    lons = [coord[0] for coord in coords]
                    lats = [coord[1] for coord in coords]
                    min_lon, max_lon = min(lons), max(lons)
                    min_lat, max_lat = min(lats), max(lats)
                    
                    print(f"Plot {plot}: bounds = ({min_lon}, {min_lat}, {max_lon}, {max_lat})")
                    
                    # Transform geographic coordinates to the orthomosaic's coordinate system
                    try:
                        # Transform bounds from WGS84 (EPSG:4326) to the orthomosaic's CRS
                        transformed_bounds = transform_bounds(
                            'EPSG:4326',  # source CRS (WGS84)
                            src.crs,      # destination CRS (orthomosaic's CRS)
                            min_lon, min_lat, max_lon, max_lat
                        )
                        min_x, min_y, max_x, max_y = transformed_bounds
                        
                        print(f"Plot {plot}: transformed bounds = ({min_x}, {min_y}, {max_x}, {max_y})")
                        
                        # Validate transformed bounds
                        if min_x >= max_x or min_y >= max_y:
                            print(f"Invalid transformed bounds for plot {plot}: min_x={min_x}, max_x={max_x}, min_y={min_y}, max_y={max_y}")
                            continue
                        
                        # Add small buffer to ensure non-zero area (in projected coordinates)
                        x_buffer = max(0.1, (max_x - min_x) * 0.1)  # 0.1 meter minimum buffer
                        y_buffer = max(0.1, (max_y - min_y) * 0.1)
                        min_x -= x_buffer
                        max_x += x_buffer
                        min_y -= y_buffer
                        max_y += y_buffer
                        
                        # Create window from transformed bounds
                        window = from_bounds(min_x, min_y, max_x, max_y, src.transform)
                        
                        print(f"Plot {plot}: window = {window} (width={window.width}, height={window.height})")
                        
                        # Validate window dimensions
                        if window.width <= 0 or window.height <= 0:
                            print(f"Invalid window dimensions for plot {plot}: width={window.width}, height={window.height}")
                            continue
                        
                        # Read the windowed data
                        data = src.read(window=window)
                        
                        # Create new transform for the windowed data
                        window_transform = src.window_transform(window)
                        
                        # Create filename
                        filename = f"plot_{plot}_accession_{accession}.tif"
                        temp_tif_path = os.path.join(output_dir, filename)
                        
                        # Write cropped TIF
                        with rasterio.open(
                            temp_tif_path,
                            'w',
                            driver='GTiff',
                            height=data.shape[1],
                            width=data.shape[2],
                            count=data.shape[0],
                            dtype=data.dtype,
                            crs=src.crs,
                            transform=window_transform,
                        ) as dst:
                            dst.write(data)
                        
                        # Convert TIF to PNG
                        png_filename = f"plot_{plot}_accession_{accession}.png"
                        png_path = os.path.join(output_dir, png_filename)
                        
                        # Open TIF and convert to PNG
                        with rasterio.open(temp_tif_path) as tif_src:
                            # Read all bands
                            data = tif_src.read()
                            
                            # Handle different band configurations
                            if data.shape[0] >= 3:  # RGB or RGBA
                                # Take first 3 bands for RGB
                                rgb_data = data[:3]
                                # Transpose from (bands, height, width) to (height, width, bands)
                                rgb_data = np.transpose(rgb_data, (1, 2, 0))
                                
                                # Normalize to 0-255 if needed
                                if rgb_data.dtype == np.uint16:
                                    rgb_data = (rgb_data / 65535.0 * 255).astype(np.uint8)
                                elif rgb_data.dtype == np.float32 or rgb_data.dtype == np.float64:
                                    rgb_data = (np.clip(rgb_data, 0, 1) * 255).astype(np.uint8)
                                
                                # Create PIL image and save as PNG
                                image = Image.fromarray(rgb_data)
                                image.save(png_path)
                                
                                plots_processed += 1
                        
                        # Remove temporary TIF file
                        os.remove(temp_tif_path)
                        
                    except Exception as e:
                        print(f"Error processing plot {plot}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error opening orthomosaic {orthomosaic_path}: {e}")
            return jsonify({"error": f"Error opening orthomosaic: {e}"}), 500
        
        return jsonify({
            "message": f"Successfully processed {plots_processed} plots",
            "plots_processed": plots_processed,
            "output_directory": output_dir
        }), 200
        
    except Exception as e:
        print(f"Error in split_orthomosaics: {e}")
        return jsonify({"error": str(e)}), 500

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/flask_app", WSGIMiddleware(file_app))

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
    file_app.config['DATA_ROOT_DIR'] = data_root_dir
    global UPLOAD_BASE_DIR
    UPLOAD_BASE_DIR = os.path.join(data_root_dir, 'Raw')

    global dir_db
    if 1:
        db_path = os.path.join(data_root_dir, "directory_index_dict.pkl")
        dir_db = None
        # Use dictionary-based index
        dir_db = DirectoryIndexDict(verbose=False)
        # Try loading from file if exists
        if os.path.exists(db_path):
            dir_db.load_dict(db_path)
            print(f"Loaded directory index dict from {db_path}")
        else:
            print(f"No dict file found, will build index from scratch.")
    else:
        db_path = os.path.join(data_root_dir, "directory_index.db")
        dir_db = None
        # Use SQLite-based index
        dir_db = DirectoryIndex(db_path=db_path, verbose=False)
        # No need to load_dict or save_dict for DirectoryIndex

    # Register inference routes
    register_inference_routes(file_app, data_root_dir)

    global now_drone_processing
    now_drone_processing = False

    # Start the Titiler server using the subprocess module
    titiler_command = f"uvicorn titiler.application.main:app --reload --port {args.titiler_port}"
    titiler_process = subprocess.Popen(titiler_command, shell=True)

    # Start the Flask server
    uvicorn.run(app, port=args.flask_port)

    # Save the directory index dict before shutdown
    if ".pkl" in db_path:
        dir_db.save_dict(db_path)

    # Terminate the Titiler server when the Flask server is shut down
    titiler_process.terminate()