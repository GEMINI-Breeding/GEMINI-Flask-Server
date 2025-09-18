# Standard library imports
import os
import json
import pandas as pd
from flask import Blueprint, jsonify, request, current_app, send_file

gcp_management_bp = Blueprint('gcp_management', __name__)

@gcp_management_bp.route('/get_gcp_selcted_images', methods=['POST'])
def get_gcp_selcted_images():
    """Get GCP selected images"""
    try:
        # Import here to avoid circular imports
        from scripts.gcp_picker import collect_gcp_candidate
        
        data = request.json
        location = data['location']
        population = data['population']
        date = data['date']
        sensor = data['sensor']
        year = data['year']
        experiment = data['experiment']
        platform = data['platform']
        
        data_root_dir = current_app.config['DATA_ROOT_DIR']
        
        result = collect_gcp_candidate(
            location=location,
            population=population, 
            date=date,
            sensor=sensor,
            year=year,
            experiment=experiment,
            platform=platform,
            data_root_dir=data_root_dir
        )
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in get_gcp_selcted_images: {e}")
        return jsonify({'error': str(e)}), 500

@gcp_management_bp.route('/refresh_gcp_selcted_images', methods=['POST'])
def refresh_gcp_selcted_images():
    """Refresh GCP selected images"""
    try:
        # Import here to avoid circular imports
        from scripts.gcp_picker import refresh_gcp_candidate
        
        data = request.json
        location = data['location']
        population = data['population']
        date = data['date']
        sensor = data['sensor']
        year = data['year']
        experiment = data['experiment']
        platform = data['platform']
        
        data_root_dir = current_app.config['DATA_ROOT_DIR']
        
        result = refresh_gcp_candidate(
            location=location,
            population=population,
            date=date,
            sensor=sensor,
            year=year,
            experiment=experiment,
            platform=platform,
            data_root_dir=data_root_dir
        )
        
        return jsonify(result)
    except Exception as e:
        print(f"Error in refresh_gcp_selcted_images: {e}")
        return jsonify({'error': str(e)}), 500

@gcp_management_bp.route('/save_array', methods=['POST'])
def save_array(debug=False):
    """Save GCP array data"""
    data = request.json
    if 'array' not in data:
        return jsonify({'error': 'Array data is required'}), 400

    # Extracting the directory path based on the first element in the array 
    base_image_path = data['array'][0]['image_path']
    platform = data['platform']
    sensor = data['sensor']
    data_root_dir = current_app.config['DATA_ROOT_DIR']
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
            for line in lines[1:]:  # Skip the first line (EPSG:4326)
                parts = line.strip().split()
                if len(parts) >= 5:
                    image_name = parts[0]
                    existing_data[image_name] = {
                        'pointX': float(parts[1]),
                        'pointY': float(parts[2]),
                        'lat': float(parts[3]),
                        'lon': float(parts[4])
                    }

    # Merge new data with existing data
    for item in data['array']:
        if 'pointX' in item and 'pointY' in item:
            image_name = os.path.basename(item['image_path'])
            existing_data[image_name] = item

    # Write merged data to file
    with open(filename, "w") as f:
        f.write('EPSG:4326\n')
        for image_name, item in existing_data.items():
            f.write(f"{image_name} {item['pointX']} {item['pointY']} {item['lat']} {item['lon']}\n")

    return jsonify({"message": f"Array saved successfully in {filename}!"}), 200

@gcp_management_bp.route('/initialize_file', methods=['POST'])
def initialize_file():
    """Initialize GCP file"""
    data = request.json
    if 'basePath' not in data:
        return jsonify({'error': 'Base path is required'}), 400

    platform = data['platform']
    sensor = data['sensor']
    data_root_dir = current_app.config['DATA_ROOT_DIR']
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
            for line in lines[1:]:  # Skip the first line (EPSG:4326)
                parts = line.strip().split()
                if len(parts) >= 5:
                    existing_data.append({
                        'image_name': parts[0],
                        'pointX': float(parts[1]),
                        'pointY': float(parts[2]),
                        'lat': float(parts[3]),
                        'lon': float(parts[4])
                    })
    else:
        # Create the file if it doesn't exist
        with open(filename, 'w') as f:
            f.write('EPSG:4326\n')

    return jsonify({"existing_data": existing_data,
                    "file_path": save_directory}), 200
