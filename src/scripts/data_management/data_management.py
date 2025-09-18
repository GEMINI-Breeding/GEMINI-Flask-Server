# Standard library imports
import os
import csv
import shutil
import pandas as pd
from flask import Blueprint, jsonify, request, current_app

data_management_bp = Blueprint('data_management', __name__)

def process_exif_data_async(file_paths, data_type, msgs_synced_file, existing_df, existing_paths):
    """Process EXIF data asynchronously"""
    # Import here to avoid import issues
    from scripts.gcp_picker import get_image_exif
    
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

@data_management_bp.route('/view_synced_data', methods=['POST'])
def view_synced_data():
    """View synchronized metadata CSV data"""
    data = request.get_json()
    base_dir = data.get('base_dir')  # Relative path like: IITA_Test/Nigeria/AmigaSample/2025-04-29/rover/RGB

    if not base_dir:
        return jsonify({'error': 'Missing base_dir'}), 400

    data_root_dir = current_app.config['DATA_ROOT_DIR']
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

@data_management_bp.route('/restore_images', methods=['POST'])
def restore_images():
    """Restore images from Removed directory back to Images directory"""
    data = request.get_json()
    image_names = data.get('images')
    removed_dir = data.get('removed_dir')  # e.g. Raw/.../Removed/top

    if not image_names or not removed_dir:
        return jsonify({'error': 'Missing images or removed_dir'}), 400

    data_root_dir = current_app.config['DATA_ROOT_DIR']
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

@data_management_bp.route('/remove_images', methods=['POST'])
def remove_images():
    """Move images from Images directory to Removed directory"""
    data = request.get_json()
    image_names = data.get('images')
    source_dir = data.get('source_dir')  # e.g. Raw/.../Images/top

    if not image_names or not source_dir:
        return jsonify({'error': 'Missing images or source_dir'}), 400

    data_root_dir = current_app.config['DATA_ROOT_DIR']
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

@data_management_bp.route('/update_data', methods=['POST'])
def update_data():
    """Update metadata and EXIF data"""
    # Import here to avoid circular imports
    from scripts.utils import process_directories_in_parallel
    
    data = request.json
    data_root_dir = current_app.config['DATA_ROOT_DIR']
    
    # Directory with image data is passed from client
    directories = data.get('directories', [])
    data_type = data.get('data_type', 'image')
    
    if not directories:
        return jsonify({'error': 'No directories provided'}), 400
    
    try:
        # Process the directories in parallel
        result = process_directories_in_parallel(
            directories, 
            data_type, 
            data_root_dir, 
            process_exif_data_async
        )
        
        return jsonify({
            'status': 'success',
            'message': 'Data updated successfully',
            'result': result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@data_management_bp.route('/best_locate_file', methods=['POST'])
def get_best_locate_file():
    """Find the best locate file for plant location results"""
    data = request.json
    location = data.get('location')
    population = data.get('population')
    date = data.get('date')
    sensor = data.get('sensor')
    year = data.get('year')
    experiment = data.get('experiment')
    platform = data.get('platform')
    
    if not all([location, population, date, sensor, year, experiment, platform]):
        return jsonify({'error': 'Missing required parameters'}), 400
    
    data_root_dir = current_app.config['DATA_ROOT_DIR']
    processed_path = os.path.join(data_root_dir, 'Processed', year, experiment, location, population, date, platform, sensor)
    
    best_file = None
    best_iteration = -1
    
    if os.path.exists(processed_path):
        for file in os.listdir(processed_path):
            if file.startswith('locate_') and file.endswith('.geojson'):
                try:
                    # Extract iteration number from filename
                    parts = file.split('_')
                    if len(parts) >= 2:
                        iteration = int(parts[1].split('.')[0])
                        if iteration > best_iteration:
                            best_iteration = iteration
                            best_file = file
                except (ValueError, IndexError):
                    continue
    
    if best_file:
        file_path = os.path.join(processed_path, best_file)
        relative_path = os.path.relpath(file_path, data_root_dir)
        return jsonify({
            'file': best_file,
            'path': relative_path,
            'iteration': best_iteration
        })
    else:
        return jsonify({'error': 'No locate files found'}), 404

@data_management_bp.route('/best_model_file', methods=['POST'])
def get_best_model_file():
    """Find the best model file"""
    data = request.json
    location = data.get('location')
    population = data.get('population')
    date = data.get('date')
    sensor = data.get('sensor')
    year = data.get('year')
    experiment = data.get('experiment')
    platform = data.get('platform')
    
    if not all([location, population, date, sensor, year, experiment, platform]):
        return jsonify({'error': 'Missing required parameters'}), 400
    
    data_root_dir = current_app.config['DATA_ROOT_DIR']
    processed_path = os.path.join(data_root_dir, 'Processed', year, experiment, location, population, date, platform, sensor)
    
    best_model = None
    best_epoch = -1
    
    if os.path.exists(processed_path):
        for file in os.listdir(processed_path):
            if file.startswith('model_epoch_') and file.endswith('.pth'):
                try:
                    # Extract epoch number from filename
                    epoch_str = file.replace('model_epoch_', '').replace('.pth', '')
                    epoch = int(epoch_str)
                    if epoch > best_epoch:
                        best_epoch = epoch
                        best_model = file
                except ValueError:
                    continue
    
    if best_model:
        file_path = os.path.join(processed_path, best_model)
        relative_path = os.path.relpath(file_path, data_root_dir)
        return jsonify({
            'file': best_model,
            'path': relative_path,
            'epoch': best_epoch
        })
    else:
        return jsonify({'error': 'No model files found'}), 404

@data_management_bp.route('/check_runs/<path:dir_path>', methods=['GET'])
def check_runs(dir_path):
    """Check for training runs in a directory"""
    data_root_dir = current_app.config['DATA_ROOT_DIR']
    full_path = os.path.join(data_root_dir, dir_path)
    
    if not os.path.exists(full_path):
        return jsonify({'runs': []}), 200
    
    try:
        runs = []
        for item in os.listdir(full_path):
            item_path = os.path.join(full_path, item)
            if os.path.isdir(item_path):
                # Check if this looks like a training run directory
                if any(f.startswith('model_epoch_') for f in os.listdir(item_path) if f.endswith('.pth')):
                    runs.append(item)
        
        runs.sort()
        return jsonify({'runs': runs})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@data_management_bp.route('/run_script', methods=['POST'])
def run_script():
    """Run a script (generic script runner)"""
    data = request.json
    script_path = data.get('script_path')
    
    if not script_path:
        return jsonify({'error': 'Script path is required'}), 400

    def run_in_thread(script_path):
        import subprocess
        try:
            result = subprocess.run(['python', script_path], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=300)  # 5 minute timeout
            print(f"Script output: {result.stdout}")
            if result.stderr:
                print(f"Script errors: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"Script {script_path} timed out")
        except Exception as e:
            print(f"Error running script {script_path}: {e}")

    import threading
    thread = threading.Thread(target=run_in_thread, args=(script_path,))
    thread.start()

    return jsonify({'status': 'Script started', 'script': script_path})
