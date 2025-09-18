# Standard library imports
import os
import shutil
import sys
import tempfile
import zipfile
import traceback
import re
import unicodedata
import threading
import pandas as pd
from multiprocessing import Process
from pathlib import Path
from flask import Blueprint, jsonify, request, current_app
from werkzeug.utils import secure_filename

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from scripts.bin_to_images.bin_to_images import extract_binary, extraction_worker
from scripts.gcp_picker import process_exif_data_async
from scripts.mavlink import process_mavlink_log_for_webapp
from scripts.create_geotiff_pyramid import create_tiled_pyramid
from scripts.directory_index import DirectoryIndex, DirectoryIndexDict  

upload_management_bp = Blueprint('upload_management', __name__)

# Global variables for upload management
extraction_processes = {}
extraction_status = "not_started"  # Possible values: not_started, in_progress, done, failed
extraction_error_message = None  # Stores detailed error message if extraction fails

def create_pyramid_external_ortho(ortho_path):
    try:
        ortho_output_path = ortho_path.replace('.tif', '-Pyramid.tif')
        create_tiled_pyramid(ortho_path, ortho_output_path)
        print(f"GeoTIFF pyramid created at {ortho_output_path}")
    except Exception as e:
        print(f"Error creating GeoTIFF pyramid: {e}")
        traceback.print_exc()

@upload_management_bp.route('/upload', methods=['POST'])
def upload_files():
    UPLOAD_BASE_DIR = current_app.config['UPLOAD_BASE_DIR']
    data_root_dir = current_app.config['DATA_ROOT_DIR']
    data_type = request.form.get('dataType')
    dir_path = request.form.get('dirPath')
    year = dir_path.split('/')[0]
    experiment = dir_path.split('/')[1]
    location = dir_path.split('/')[2]
    population = dir_path.split('/')[3]
    date = dir_path.split('/')[4]
    platform = dir_path.split('/')[5] if len(dir_path.split('/')) > 5 else ''
    sensor = dir_path.split('/')[6] if len(dir_path.split('/')) > 6 else ''
    upload_new_files_only = request.form.get('uploadNewFilesOnly') == 'true'

    if data_type == 'ortho':
        full_dir_path = os.path.join(data_root_dir, 'Processed', year, experiment, location, population, date, platform, sensor)
    else:
        # Sanitize the directory path to remove any hidden Unicode characters    
        # Normalize Unicode and remove control characters
        dir_path_clean = unicodedata.normalize('NFKD', dir_path)
        dir_path_clean = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', dir_path_clean)  # Remove control characters
        dir_path_clean = re.sub(r'[^\x20-\x7e]', '', dir_path_clean)  # Keep only ASCII printable characters
        dir_path_clean = dir_path_clean.strip()  # Remove leading/trailing whitespace
        full_dir_path = os.path.join(UPLOAD_BASE_DIR, dir_path_clean)
        if dir_path != dir_path_clean:
            print(f"Original dir_path: {repr(dir_path)}")
            print(f"Cleaned dir_path: {repr(dir_path_clean)}")
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
        elif data_type == 'ortho':
            if filename.endswith('DEM.tif'):
                filename = f'{date}-DEM.tif'
            elif filename.endswith('RGB.tif'):
                filename = f'{date}-RGB.tif'
        file_path = os.path.join(full_dir_path, filename)
        if upload_new_files_only and os.path.isfile(file_path):
            print(f"Skipping {filename} because it already exists in {dir_path}")
        else:
            file.save(file_path)
            if filename.endswith('DEM.tif') or filename.endswith('RGB.tif'):
                create_pyramid_external_ortho(file_path)
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
    dir_db = current_app.config['DIR_DB']
    if uploaded_file_paths and dir_db is not None:
        try:
            # Refresh the directory in the database
            dir_db.force_refresh(full_dir_path)
            print(f"Updated directory database for: {full_dir_path}")
        except Exception as e:
            print(f"Error updating directory database: {e}")

    return jsonify({'message': 'Files uploaded successfully'}), 200

@upload_management_bp.route('/check_files', methods=['POST'])
def check_files():
    data = request.json
    fileList = data['fileList']
    dirPath = data['dirPath']
    UPLOAD_BASE_DIR = current_app.config['UPLOAD_BASE_DIR']
    full_dir_path = os.path.join(UPLOAD_BASE_DIR, dirPath)

    existing_files = set(os.listdir(full_dir_path)) if os.path.exists(full_dir_path) else set()
    new_files = [file for file in fileList if file not in existing_files]

    print(f"Uploading {str(len(new_files))} out of {str(len(fileList))} files to {dirPath}")

    return jsonify(new_files), 200

@upload_management_bp.route("/get_binary_report", methods=["POST"])
def get_binary_report():
    data = request.json
    # Construct file path based on metadata
    UPLOAD_BASE_DIR = current_app.config['UPLOAD_BASE_DIR']
    report_path = f"{UPLOAD_BASE_DIR}/{data['year']}/{data['experiment']}/{data['location']}/{data['population']}/{data['date']}/rover/RGB/report.txt"
    if not os.path.exists(report_path):
        report_path = f"{UPLOAD_BASE_DIR}/{data['year']}/{data['experiment']}/{data['location']}/{data['population']}/{data['date']}/Amiga/RGB/report.txt"
    try:
        with open(report_path, "r") as f:
            content = f.read()
        return content, 200
    except Exception as e:
        return f"Error loading report: {str(e)}", 500

@upload_management_bp.route('/cancel_extraction', methods=['POST'])
def cancel_extraction():
    data = request.json
    dir_path = data.get('dirPath')
    p = extraction_processes.pop(dir_path, None)
    if not p:
        return jsonify({'status': 'no active extraction for this path'}), 404

    p.terminate()  # force-kill the worker
    p.join()
    return jsonify({'status': 'cancelled'}), 200


@upload_management_bp.route('/get_binary_status', methods=['GET'])
def get_binary_status():
    print(f"Extraction status: {extraction_status}")
    response = {'status': extraction_status}
    if extraction_status == "failed" and extraction_error_message:
        response['error_message'] = extraction_error_message
    return jsonify(response), 200

@upload_management_bp.route('/extract_binary_file', methods=['POST'])
def extract_binary_file():
    UPLOAD_BASE_DIR = current_app.config['UPLOAD_BASE_DIR']
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

@upload_management_bp.route('/get_binary_progress', methods=['POST'])
def get_binary_progress():
    UPLOAD_BASE_DIR = current_app.config['UPLOAD_BASE_DIR']
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

@upload_management_bp.route('/upload_chunk', methods=['POST'])
def upload_chunk():
    UPLOAD_BASE_DIR = current_app.config['UPLOAD_BASE_DIR']
    data_root_dir = current_app.config['DATA_ROOT_DIR']
    dir_db = current_app.config['DIR_DB']
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
    if file_name.endswith('-RGB.tif') or file_name.endswith('-DEM.tif'):
        full_dir_path = os.path.join(data_root_dir, dir_path_clean)
    else:
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
            if file_name.endswith('DEM.tif') or file_name.endswith('RGB.tif'):
                file_path = os.path.join(full_dir_path, file_name)
                create_pyramid_external_ortho(file_path)
                full_raw_path = full_dir_path.replace('Processed/', 'Raw/')
                os.makedirs(full_raw_path, exist_ok=True)
            return "File reassembled and saved successfully", 200
        except Exception as e:
            print(f"Error during reassembly: {e}")
            return f"Error during reassembly: {e}", 500
    else:
        return f"Chunk {chunk_index} of {total_chunks} received", 202
    
@upload_management_bp.route('/check_uploaded_chunks', methods=['POST'])
def check_uploaded_chunks():
    UPLOAD_BASE_DIR = current_app.config['UPLOAD_BASE_DIR']
    data = request.json
    file_identifier = data['fileIdentifier']
    dir_path = data['localDirPath']
    full_dir_path = os.path.join(UPLOAD_BASE_DIR, dir_path)
    cache_dir_path = os.path.join(full_dir_path, 'cache')
    
    uploaded_chunks = [f for f in os.listdir(cache_dir_path) if f.startswith(file_identifier)]
    uploaded_chunks_count = len(uploaded_chunks)

    return jsonify({'uploadedChunksCount': uploaded_chunks_count}), 200

@upload_management_bp.route('/clear_upload_dir', methods=['POST'])
def clear_upload_dir():
    UPLOAD_BASE_DIR = current_app.config['UPLOAD_BASE_DIR']
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

@upload_management_bp.route('/clear_upload_cache', methods=['POST'])
def clear_upload_cache():
    UPLOAD_BASE_DIR = current_app.config['UPLOAD_BASE_DIR']
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