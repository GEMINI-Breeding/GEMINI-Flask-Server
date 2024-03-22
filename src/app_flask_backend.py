from concurrent.futures import thread
from enum import unique
from math import e
import os
import re
import subprocess
import threading
import uvicorn
import signal
import flask
import time
import glob
import yaml
import random
import string
import shutil
import asyncio

from flask import Flask, send_from_directory, jsonify, request, send_file
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor, as_completed
from subprocess import CalledProcessError
import shutil

import csv
import json
import traceback
from PIL import Image
from pyproj import Geod
import pandas as pd
import geopandas as gpd

from werkzeug.utils import secure_filename
from pathlib import Path
import concurrent.futures

import argparse
from scripts.drone_trait_extraction.drone_gis import process_tiff, find_drone_tiffs
from scripts.orthomosaic_generation import run_odm

# Define the Flask application for serving files
file_app = Flask(__name__)
latest_data = {'epoch': 0, 'map': 0, 'locate': 0, 'extract': 0, 'ortho': 0}
training_stopped_event = threading.Event()

#### FILE SERVING ENDPOINTS ####
# endpoint to serve files
@file_app.route('/files/<path:filename>')
def serve_files(filename):
    global data_root_dir
    return send_from_directory(data_root_dir, filename)

# endpoint to list directories
@file_app.route('/list_dirs/<path:dir_path>', methods=['GET'])
def list_dirs(dir_path):
    global data_root_dir
    dir_path = os.path.join(data_root_dir, dir_path)  # join with base directory path
    if os.path.exists(dir_path):
        dirs = next(os.walk(dir_path))[1]
        return jsonify(dirs), 200
    else:
        return jsonify({'message': 'Directory not found'}), 404
    
def build_nested_structure_sync(path, current_depth=0, max_depth=2):
    if current_depth >= max_depth:
        return {}
    
    structure = {}
    for child in path.iterdir():
        if child.is_dir():
            structure[child.name] = build_nested_structure_sync(child, current_depth+1, max_depth)
    return structure

async def build_nested_structure(path, current_depth=0, max_depth=2):
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, build_nested_structure_sync, path, current_depth, max_depth)

async def process_directories_in_parallel(base_dir, max_depth=2):
    directories = [d for d in base_dir.iterdir() if d.is_dir()]
    tasks = [build_nested_structure(d, 0, max_depth) for d in directories]
    nested_structures = await asyncio.gather(*tasks)
    
    combined_structure = {}
    for d, structure in zip(directories, nested_structures):
        combined_structure[d.name] = structure
    
    return combined_structure

@file_app.get("/list_dirs_nested")
async def list_dirs_nested():
    base_dir = Path(data_root_dir) / 'Raw'
    combined_structure = await process_directories_in_parallel(base_dir, max_depth=7)
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
    
@file_app.route('/check_runs/<path:dir_path>', methods=['GET'])
def check_runs(dir_path):
    global data_root_dir
    dir_path = os.path.join(data_root_dir, dir_path)
    response_data = {}  # Initialize an empty dictionary for the response
    
    # For the Model column of Locate Plants
    if os.path.exists(dir_path) and 'Plant Detection' in dir_path:
        check = f'{dir_path}/Plant-*/weights/best.pt'
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
        with open(logs, 'r') as file:
            data = yaml.safe_load(file)
            response_data = {k: {'model': v['model'], 'locate': v['locate'], 'id': v['id']} for k, v in data.items()}
        
    return jsonify(response_data), 200
    
@file_app.route('/upload', methods=['POST'])
def upload_files():
    dir_path = request.form.get('dirPath')
    upload_new_files_only = request.form.get('uploadNewFilesOnly') == 'true'
    full_dir_path = os.path.join(UPLOAD_BASE_DIR, dir_path)
    os.makedirs(full_dir_path, exist_ok=True)

    for file in request.files.getlist("files"):
        filename = secure_filename(file.filename)
        file_path = os.path.join(full_dir_path, filename)

        if upload_new_files_only and os.path.isfile(file_path):
            print(f"Skipping {filename} because it already exists in {dir_path}")
            continue  # Skip existing file

        file.save(file_path)

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
    if all(os.path.exists(os.path.join(cache_dir_path, f"{file_name}.part{i}")) for i in range(int(total_chunks))):
        # Reassemble file
        with open(os.path.join(full_dir_path, file_name), 'wb') as full_file:
            for i in range(int(total_chunks)):
                with open(os.path.join(cache_dir_path, f"{file_name}.part{i}"), 'rb') as part_file:
                    full_file.write(part_file.read())
                os.remove(os.path.join(cache_dir_path, f"{file_name}.part{i}"))  # Clean up chunk

        os.remove(os.path.join(full_dir_path, 'cache'))  # Clean up cache directory
        return "File reassembled and saved successfully", 200
    else:
        return f"Chunk {chunk_index} of {total_chunks} received", 202
    
@file_app.route('/check_uploaded_chunks', methods=['POST'])
def check_uploaded_chunks():
    data = request.json
    file_identifier = data['fileIdentifier']
    dir_path = data['dirPath']
    full_dir_path = os.path.join(UPLOAD_BASE_DIR, dir_path)
    cache_dir_path = os.path.join(full_dir_path, 'cache')
    
    uploaded_chunks = [f for f in os.listdir(cache_dir_path) if f.startswith(file_identifier)]
    uploaded_chunks_count = len(uploaded_chunks)

    return jsonify({'uploadedChunksCount': uploaded_chunks_count}), 200

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

#### IMAGE PROCESSING ENDPOINTS ####
# Function to calculate the distance between two coordinates using pyproj
def calculate_distance(lat1, lon1, lat2, lon2):
    geod = Geod(ellps='WGS84')
    _, _, distance = geod.inv(lon1, lat1, lon2, lat2)
    return distance

@file_app.route('/process_images', methods=['POST'])
def process_images():
    global data_root_dir
    # receive the parameters
    location = request.json['location']
    population = request.json['population']
    date = request.json['date']
    radius_meters = request.json['radius_meters']
    year = request.json['year']
    experiment = request.json['experiment']
    sensor = request.json['sensor']
    platform = request.json['platform']

    prefix = data_root_dir+'/Raw'
    image_folder = os.path.join(prefix, year, experiment, location, population, date, platform, sensor, 'Images')

    print("Loading predefined locations from CSV file...")

    # Define the path to the predefined locations CSV file
    predefined_locations_csv = os.path.join(prefix, year, experiment, location, population, 'gcp_locations.csv')

    # Load predefined locations from CSV
    if not os.path.isfile(predefined_locations_csv):
        raise Exception("Invalid selections: no gcp_locations.csv file found.")

    df = pd.read_csv(predefined_locations_csv)
    labels = df['Label'].tolist()
    latitudes = df['Lat_dec'].tolist()
    longitudes = df['Lon_dec'].tolist()
    predefined_locations = []
    for i in range(len(labels)):
        predefined_locations.append({
            'label': labels[i],
            'latitude': latitudes[i],
            'longitude': longitudes[i]
        })

    # Select the image folder
    if not os.path.isdir(image_folder):
        raise Exception("Invalid selections: no image folder.")

    selected_images = []

    # Process each image in the folder
    files = os.listdir(image_folder)
    files.sort()

    if len(files) == 0:
        raise Exception("Invalid selections: no files found in folder.")

    for filename in files:
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            # print("Processing image: " + filename)
            image_path = os.path.join(image_folder, filename)

            # Extract GPS coordinates from EXIF data
            image = Image.open(image_path)

            # Get image dimensions
            width, height = image.size

            exif_data = image._getexif()
            if exif_data is not None and 34853 in exif_data:
                gps_info = exif_data[34853]
                latitude = gps_info[2][0] + gps_info[2][1] / 60 + gps_info[2][2] / 3600
                longitude = gps_info[4][0] + gps_info[4][1] / 60 + gps_info[4][2] / 3600
                latitude = float(latitude)
                longitude = float(longitude) * -1

                # Check if the image is within the predefined locations
                closest_dist = float('inf')
                closest_location = None
                for location in predefined_locations:
                    dist = calculate_distance(latitude, longitude, location['latitude'], location['longitude'])
                    if dist <= radius_meters and dist < closest_dist:
                        closest_dist = dist
                        closest_location = location

                if closest_location is not None:

                    # Remove the first part of the image path
                    image_path = image_path.replace(data_root_dir, '')

                    selected_images.append({
                        'image_path': image_path,
                        'gcp_lat': closest_location['latitude'],
                        'gcp_lon': closest_location['longitude'],
                        'gcp_label': closest_location['label'],
                        'naturalWidth': width,
                        'naturalHeight': height
                    })

    # Return the selected images and their corresponding GPS coordinates
    return jsonify({'selected_images': selected_images,
                    'num_total': len(files)}), 200


@file_app.route('/process_drone_tiff', methods=['POST'])
def process_drone_tiff(dir_path):
    # receive the parameters
    location = request.json['location']
    population = request.json['population']
    date = request.json['date']
    year = request.json['year']
    experimnent = request.json['experiment']

    prefix = data_root_dir+'/Processed'
    image_folder = os.path.join(prefix, year, experimnent, location, population, date, 'Drone')
    dir_path = os.path.join(prefix, year, experimnent, location, population, date)

    # Check if already in processing
    global now_drone_processing
    if now_drone_processing:
        return jsonify({'message': 'Already in processing'}), 400
    
    now_drone_processing = True
    
    try: 
        rgb_tif_file, dem_tif_file, thermal_tif_file = find_drone_tiffs(image_folder)
        geojson_path = os.path.join(dir_path,'Plot-Attributes-WGS84.geojson')
        date = dir_path.split("/")[-1]
        sensor = "Drone"
        output_geojson = os.path.join(os.path.dirname(image_folder),"Results",f"{date}-{sensor}-Traits-WGS84.geojson")
        process_tiff(tiff_files_rgb=os.path.join(image_folder,rgb_tif_file),
                     tiff_files_dem=os.path.join(image_folder,dem_tif_file),
                     tiff_files_thermal=os.path.join(image_folder,thermal_tif_file),
                     plot_geojson=geojson_path,
                     output_geojson=output_geojson,
                     debug=False)


    except Exception as e:
        now_drone_processing = False
        return jsonify({'message': str(e)}), 400



    now_drone_processing = False

    return jsonify({'message': 'Processing Finished'}), 200



@file_app.route('/save_array', methods=['POST'])
def save_array():
    data = request.json
    if 'array' not in data:
        return jsonify({"message": "Missing array in data"}), 400

    # Extracting the directory path based on the first element in the array 
    base_image_path = data['array'][0]['image_path']
    platform = data['platform']
    sensor = data['sensor']
    processed_path = os.path.join(base_image_path.replace('/Raw/', 'Intermediate/').split(f'/{platform}')[0], platform, sensor)
    save_directory = os.path.join(data_root_dir, processed_path)
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

def filter_images(geojson_features, year, experiment, location, population, date, sensor, middle_image=False):

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

    filtered_images = [{'imageName': image, 'label': label, 'plot': plot} for image, label, plot in zip(filtered_images, filtered_labels, filtered_plots)]

    # Sort the filtered_images by label
    filtered_images = sorted(filtered_images, key=lambda x: x['label'])

    return filtered_images

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

    filtered_images = filter_images(geojson_features, year, experiment, location, 
                                    population, date, sensor, middle_image)

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
    args = argparse.Namespace()
    args.data_root_dir = data_root_dir
    args.location = location
    args.population = population
    args.date = date
    args.year = year
    args.experiment = experiment
    args.platform = platform
    args.sensor = sensor
    args.temp_dir = temp_dir
    args.reconstruction_quality = reconstruction_quality
    args.custom_options = custom_options
    
    try:
        # Reset ODM
        reset_odm()
        
        # Run ODM in a separate thread
        thread = threading.Thread(target=run_odm, args=(args,))
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
        return make_response(jsonify({"status": "error", "message": f"ODM processing failed to start {str(e)}"}), 400)

def reset_odm():
    # Delete existing folders
    temp_path = os.path.join(data_root_dir, 'temp')
    while os.path.exists(temp_path):
        shutil.rmtree(temp_path)
        
@file_app.route('/stop_odm', methods=['POST'])
def stop_odm():
    try:
        print('ODM processed stopped by user.')
        stop_event = threading.Event()
        stop_event.set()
        reset_odm()
        return jsonify({"message": "ODM process stopped"}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": e.stderr.decode("utf-8")}), 500

@file_app.route('/get_ortho_progress', methods=['GET'])
def get_ortho_progress():
    return jsonify(latest_data)

def update_progress_file(progress_file, progress):
    with open(progress_file, 'w') as pf:
        pf.write(f"{progress}%")
        latest_data['ortho'] = progress
        print('Ortho progress updated:', progress)
               
def monitor_log_updates(logs_path, progress_file):
    
    try:
        progress_points = [
            "Running opensfm stage",
            "Export reconstruction stats",
            "Estimated depth-maps",
            "Geometric-consistent estimated depth-maps",
            "Filtered depth-maps",
            "Fused depth-maps",
            "Point visibility checks",
            "Decimated faces",
            "Running odm_georeferencing stage",
            "running pdal translate"
        ]
        
        completed_stages = set()
        progress_increment = 10  # Each stage completion increases progress by 10%
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
            
            while True:
                line = file.readline()
                if line:
                    for point in progress_points:
                        if point in line and point not in completed_stages:
                            completed_stages.add(point)
                            current_progress = len(completed_stages) * progress_increment
                            update_progress_file(progress_file, current_progress)
                            print(f"Progress updated: {current_progress}%")
                else:
                    time.sleep(1)  # Sleep briefly to avoid busy waiting
    except Exception as e:
        # Handle exception: log it, set a flag, etc.
        print(f"Error in thread: {e}")

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
        
    except Exception as e:
        print(f'Error preparing labels for training: {e}')

### ROVER MODEL TRAINING ###
def check_model_details(key):
    
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
    mAP = round(df['metrics/mAP50(B)'].max(), 2)
    values.extend([mAP])
    
    # get run name
    run = base_path.name
    match = re.search(r'-([A-Za-z0-9]+)$', run)
    id = match.group(1)
    
    # collate details
    details = {'id': id, 'epochs': epochs, 'batch': batch, 'imgsz': imgsz, 'map': mAP}
    
    return details

@file_app.route('/get_model_info', methods=['POST'])
def get_model_info():
    data = request.json
    details_data = []
    
    # iterate through each existing model
    for key in data:
        details = check_model_details(Path(key))
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
    print(latest_data)
    return jsonify(latest_data)

@file_app.route('/train_model', methods=['POST'])
def train_model():
    global data_root_dir, latest_data, training_stopped_event, new_folder, train_labels
    
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
        prepare_labels(annotations, all_images)
        
        # extract labels
        labels_path = Path(data_root_dir)/'Intermediate'/year/experiment/location/population/date/platform/sensor/f'Labels/{trait} Detection/labels/train'
        labels = get_labels(labels_path)
        labels_arg = " ".join(labels)
        
        # other training args
        container_dir = Path('/app/mnt/GEMINI-App-Data')
        pretrained = "/app/train/yolov8n.pt"
        save_train_model = container_dir/'Intermediate'/year/experiment/location/population/'Training'/platform
        scan_save = Path(data_root_dir)/'Intermediate'/year/experiment/location/population/'Training'/platform/f'{sensor} {trait} Detection'
        scan_save = Path(scan_save)
        scan_save.mkdir(parents=True, exist_ok=True)
        latest_data['epoch'] = 0
        latest_data['map'] = 0
        training_stopped_event.clear()
        threading.Thread(target=scan_for_new_folders, args=(scan_save,), daemon=True).start()
        images = container_dir/'Intermediate'/year/experiment/location/population/date/platform/sensor/f'Labels/{trait} Detection'
        
        # run training
        cmd = (f"docker exec train "
            f"python /app/train/train.py "
            f"--pretrained '{pretrained}' --images '{images}' --save '{save_train_model}' --sensor '{sensor}' "
            f"--date '{date}' --trait '{trait}' --image-size '{image_size}' --epochs '{epochs}' "
            f"--batch-size {batch_size} --labels {labels_arg} ")
        
        print(cmd)

        process = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = process.stdout.decode('utf-8')
        # output = 'test'
        return jsonify({"message": "Training started", "output": output}), 202
    except subprocess.CalledProcessError as e:
        error_output = e.stderr.decode('utf-8')
        return jsonify({"error": error_output}), 500
    
@file_app.route('/stop_training', methods=['POST'])
def stop_training():
    global training_stopped_event, new_folder, labels_train_folder, labels_val_folder, images_train_folder, images_val_folder
    container_name = 'train'
    try:        
        # stop training
        print('Training stopped by user.')
        kill_cmd = f"docker exec {container_name} pkill -9 -f python"
        subprocess.run(kill_cmd, shell=True)
        print(f"Sent SIGKILL to Python process in {container_name} container.")
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
    global training_stopped_event, new_folder, results_file, labels_train_folder, labels_val_folder, images_train_folder, images_val_folder
    container_name = 'train'
    try:
        # stop training
        print('Training stopped by user.')
        kill_cmd = f"docker exec {container_name} pkill -9 -f python"
        subprocess.run(kill_cmd, shell=True)
        print(f"Sent SIGKILL to Python process in {container_name} container.")
        training_stopped_event.set()
        results_file = ''
        
        # unlink files
        remove_files_from_folder(labels_train_folder)
        remove_files_from_folder(labels_val_folder)
        remove_files_from_folder(images_train_folder)
        remove_files_from_folder(images_val_folder)
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
    
    # collate details
    details = {'id': id, 'model': model_id, 'count': stand_count}
    
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
    global data_root_dir, save_locate
    
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
    container_dir = Path('/app/mnt/GEMINI-App-Data')
    images = container_dir/'Raw'/year/experiment/location/population/date/platform/sensor/'Images'
    disparity = Path(container_dir/'Raw'/year/experiment/location/population/date/platform/sensor/'Disparity')
    configs = container_dir/'Raw'/year/experiment/location/population/date/platform/sensor/'Metadata'
    plotmap = container_dir/'Intermediate'/year/experiment/location/population/'Plot-Attributes-WGS84.geojson'
    
    # generate save folder
    version = generate_hash(trait='Locate')
    save_base = Path(data_root_dir)/'Intermediate'/year/experiment/location/population/date/platform/sensor/f'Locate'
    while (save_base / f'{version}').exists():
        version = generate_hash(trait='Locate')
    save_locate = Path(data_root_dir)/'Intermediate'/year/experiment/location/population/date/platform/sensor/f'Locate'/f'{version}'
    save_locate.mkdir(parents=True, exist_ok=True)
    save = Path(container_dir/'Intermediate'/year/experiment/location/population/date/platform/sensor/f'Locate'/f'{version}')
    model_path = container_dir/'Intermediate'/year/experiment/location/population/'Training'/f'{platform}'/'RGB Plant Detection'/f'Plant-{id}'/'weights'/'last.pt'
    
    # save logs file
    data = {"model": [id], "date": [date]}
    with open(save_locate/"logs.yaml", "w") as file:
        yaml.dump(data, file, default_flow_style=False)
        
    # create progress file
    with open(save_locate/"locate_progress.txt", "w") as file:
        pass
    
    # run locate
    if disparity.exists():
        cmd = (
        f"docker exec locate-extract "
        f"python -W ignore /app/locate.py "
        f"--images '{images}' --metadata '{configs}' --plotmap '{plotmap}' "
        f"--batch-size '{batch_size}' --model '{model_path}' --save '{save}' --skip-stereo"
        )
    else:   
        cmd = (
        f"docker exec locate-extract "
        f"python -W ignore /app/locate.py "
        f"--images '{images}' --metadata '{configs}' --plotmap '{plotmap}' "
        f"--batch-size '{batch_size}' --model '{model_path}' --save '{save}' "
        )
    print(cmd)

    try:
        process = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = process.stdout.decode('utf-8')
        return jsonify({"message": "Locate has started", "output": output}), 202
    except subprocess.CalledProcessError as e:
        error_output = e.stderr.decode('utf-8')
        return jsonify({"error": error_output}), 500
    
@file_app.route('/stop_locate', methods=['POST'])
def stop_locate():
    global save_locate
    container_name = 'locate-extract'
    try:
        print('Locate-Extract stopped by user.')
        kill_cmd = f"docker exec {container_name} pkill -9 -f python"
        subprocess.run(kill_cmd, shell=True)
        print(f"Sent SIGKILL to Python process in {container_name} container.")
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
    global data_root_dir, save_extract, temp_extract, model_id, summary_date, locate_id, trait_extract
    
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
        container_dir = Path('/app/mnt/GEMINI-App-Data')
        summary_path = container_dir/'Intermediate'/year/experiment/location/population/summary_date/platform/sensor/'Locate'/f'Locate-{locate_id}'/'locate.csv'
        model_path = container_dir/'Intermediate'/year/experiment/location/population/'Training'/platform/f'RGB {trait} Detection'/f'{trait}-{model_id}'/'weights'/'best.pt'
        images = container_dir/'Raw'/year/experiment/location/population/date/platform/sensor/'Images'
        disparity = Path(container_dir/'Raw'/year/experiment/location/population/date/platform/sensor/'Disparity')
        plotmap = container_dir/'Intermediate'/year/experiment/location/population/'Plot-Attributes-WGS84.geojson'
        metadata = container_dir/'Raw'/year/experiment/location/population/date/platform/sensor/'Metadata'
        save = container_dir/'Processed'/year/experiment/location/population/date/platform/sensor/f'{date}-{platform}-{sensor}-{trait}-Traits-WGS84.geojson'
        save_extract = Path(data_root_dir)/'Processed'/year/experiment/location/population/date/platform/sensor
        temp = container_dir/'Processed'/year/experiment/location/population/date/platform/sensor/'temp'
        temp_extract = Path(data_root_dir)/'Processed'/year/experiment/location/population/date/platform/sensor/'temp'
        temp_extract.mkdir(parents=True, exist_ok=True) #if it doesnt exists
        save_extract.mkdir(parents=True, exist_ok=True)
        
        # check if date is emerging
        emerging = date in summary
        
        # run extract
        if emerging:
            if disparity.exists():
                cmd = (
                    f"docker exec locate-extract /bin/sh -c \""
                    f". /miniconda/etc/profile.d/conda.sh && "
                    f"conda activate env && "
                    f"exec python -W ignore /app/extract.py "
                    f"--emerging --summary '{summary_path}' --images '{images}' --plotmap '{plotmap}' "
                    f"--batch-size {batch_size} --model-path '{model_path}' --save '{save}' "
                    f"--metadata '{metadata}' --temp '{temp}' --trait '{trait}' --skip-stereo\""
                )
            else:
                cmd = (
                    f"docker exec locate-extract /bin/sh -c \""
                    f". /miniconda/etc/profile.d/conda.sh && "
                    f"conda activate env && "
                    f"exec python -W ignore /app/extract.py "
                    f"--emerging --summary '{summary_path}' --images '{images}' --plotmap '{plotmap}' "
                    f"--batch-size {batch_size} --model-path '{model_path}' --save '{save}' "
                    f"--metadata '{metadata}' --temp '{temp}' --trait '{trait}'\""
                )
        else:
            if disparity.exists():
                cmd = (
                    f"docker exec locate-extract /bin/sh -c \""
                    f". /miniconda/etc/profile.d/conda.sh && "
                    f"conda activate env && "
                    f"exec python -W ignore /app/extract.py "
                    f"--summary '{summary_path}' --images '{images}' --plotmap '{plotmap}' "
                    f"--batch-size {batch_size} --model-path '{model_path}' --save '{save}' "
                    f"--metadata '{metadata}' --temp '{temp}' --trait '{trait}' --skip-stereo\""
                )
            else:
                cmd = (
                    f"docker exec locate-extract /bin/sh -c \""
                    f". /miniconda/etc/profile.d/conda.sh && "
                    f"conda activate env && "
                    f"exec python -W ignore /app/extract.py "
                    f"--summary '{summary_path}' --images '{images}' --plotmap '{plotmap}' "
                    f"--batch-size {batch_size} --model-path '{model_path}' --save '{save}' "
                    f"--metadata '{metadata}' --temp '{temp}' --trait '{trait}' --skip-stereo\""
                )
        print(cmd)
        
        process = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = process.stdout.decode('utf-8')
        return jsonify({"message": "Extract has started", "output": output}), 202
    
    except subprocess.CalledProcessError as e:
        error_output = e.stderr.decode('utf-8')
        return jsonify({"status": "error", "message": str(error_output)}), 400
    
@file_app.route('/stop_extract', methods=['POST'])
def stop_extract():
    global save_extract, temp_extract
    container_name = 'locate-extract'
    try:
        print('Locate-Extract stopped by user.')
        kill_cmd = f"docker exec {container_name} pkill -9 -f python"
        subprocess.run(kill_cmd, shell=True)
        print(f"Sent SIGKILL to Python process in {container_name} container.")
        subprocess.run(f"rm -rf '{save_extract}/logs.yaml'", check=True, shell=True)
        subprocess.run(f"rm -rf '{temp_extract}'", check=True, shell=True)
        return jsonify({"message": "Python process in container successfully stopped"}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": e.stderr.decode("utf-8")}), 500
    
@file_app.route('/done_extract', methods=['POST'])
def done_extract():
    global temp_extract, save_extract, model_id, summary_date, locate_id, trait_extract
    container_name = 'locate-extract'
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
        
        print('Training stopped by user.')
        kill_cmd = f"docker exec {container_name} pkill -9 -f python"
        subprocess.run(kill_cmd, shell=True)
        print(f"Sent SIGKILL to Python process in {container_name} container.")
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
    parser.add_argument('--data_root_dir', type=str, default='/home/GEMINI/GEMINI-App-Data',required=False)
    parser.add_argument('--port', type=int, default=5000,required=False) # Default port is 5000
    args = parser.parse_args()

    # Print the arguments to the console
    print(f"data_root_dir: {args.data_root_dir}")
    print(f"port: {args.port}")

    # Update global data_root_dir from the argument
    global data_root_dir
    data_root_dir = args.data_root_dir

    UPLOAD_BASE_DIR = os.path.join(data_root_dir, 'Raw')

    global now_drone_processing
    now_drone_processing = False

    # Start the Titiler server using the subprocess module
    titiler_command = "uvicorn titiler.application.main:app --reload --port 8090"
    titiler_process = subprocess.Popen(titiler_command, shell=True)

    # Start the Flask server
    uvicorn.run(app, host="127.0.0.1", port=args.port)

    # Terminate the Titiler server when the Flask server is shut down
    titiler_process.terminate()
