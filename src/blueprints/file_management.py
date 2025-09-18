# Standard library imports
import os
import rasterio
import asyncio
from pathlib import Path
from flask import Blueprint, send_from_directory, jsonify, request, current_app, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import io
import yaml
import csv
import re
import shutil
from scripts.utils import process_directories_in_parallel, process_directories_in_parallel_from_db
from scripts.orthomosaic_generation import convert_tif_to_png

file_management_bp = Blueprint('file_management', __name__)

@file_management_bp.route('/get_tif_to_png', methods=['POST'])
def get_tif_to_png():
    data = request.json
    tif_path = data['filePath']
    data_root_dir = current_app.config['DATA_ROOT_DIR']
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
@file_management_bp.route('/files/<path:filename>')
def serve_files(filename):
    data_root_dir = current_app.config['DATA_ROOT_DIR']
    return send_from_directory(data_root_dir, filename)

# endpoint to serve image in memory
@file_management_bp.route('/images/<path:filename>')
def serve_image(filename):
    global image_dict
    return image_dict[filename]

# endpoint to serve PNG files directly
@file_management_bp.route('/get_png_file', methods=['POST'])
def get_png_file():
    data_root_dir = current_app.config['DATA_ROOT_DIR']
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

@file_management_bp.route('/fetch_data_root_dir', methods=['GET'])
def fetch_data_root_dir():
    data_root_dir = current_app.config['DATA_ROOT_DIR']
    return data_root_dir

# endpoint to list directories
@file_management_bp.route('/list_dirs/<path:dir_path>', methods=['GET'])
def list_dirs(dir_path):
    """Fast directory listing using index"""
    data_root_dir = current_app.config['DATA_ROOT_DIR']
    dir_db = current_app.config['DIR_DB']

    full_path = os.path.join(data_root_dir, dir_path)

    # Try index first
    dirs = dir_db.get_children(full_path, directories_only=True, wait_if_needed=True)

    return jsonify(dirs), 200

@file_management_bp.get("/list_dirs_nested")
async def list_dirs_nested():
    data_root_dir = current_app.config['DATA_ROOT_DIR']
    dir_db = current_app.config['DIR_DB']
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

@file_management_bp.get("/list_dirs_nested_processed")
async def list_dirs_nested_processed():
    data_root_dir = current_app.config['DATA_ROOT_DIR']
    dir_db = current_app.config['DIR_DB']
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
@file_management_bp.route('/list_files/<path:dir_path>', methods=['GET'])
def list_files(dir_path):
    """Fast file listing using directory index"""
    data_root_dir = current_app.config['DATA_ROOT_DIR']
    dir_db = current_app.config['DIR_DB']
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

@file_management_bp.route('/view_synced_data', methods=['POST'])
def view_synced_data():
    data = request.get_json()
    base_dir = data.get('base_dir')  # Relative path like: IITA_Test/Nigeria/AmigaSample/2025-04-29/rover/RGB
    data_root_dir = current_app.config['DATA_ROOT_DIR']
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

@file_management_bp.route('/restore_images', methods=['POST'])
def restore_images():
    data_root_dir = current_app.config['DATA_ROOT_DIR']
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

@file_management_bp.route('/remove_images', methods=['POST'])
def remove_images():
    data_root_dir = current_app.config['DATA_ROOT_DIR']

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

@file_management_bp.route('/update_data', methods=['POST'])
def update_data():
    data_root_dir = current_app.config['DATA_ROOT_DIR']
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

@file_management_bp.route('/delete_files', methods=['POST'])
def delete_files():
    data_root_dir = current_app.config['DATA_ROOT_DIR']
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
    
@file_management_bp.route('/best_locate_file', methods=['POST'])
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

@file_management_bp.route('/best_model_file', methods=['POST'])
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

@file_management_bp.route('/check_runs/<path:dir_path>', methods=['GET'])
def check_runs(dir_path):
    data_root_dir = current_app.config['DATA_ROOT_DIR']
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
    