from concurrent.futures import thread
from math import e
import os
import re
import subprocess
import threading
import uvicorn
import time

from flask import Flask, send_from_directory, jsonify, request
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.middleware.cors import CORSMiddleware

import csv
import json
from PIL import Image
from pyproj import Geod
import pandas as pd
import geopandas as gpd

from werkzeug.utils import secure_filename
from pathlib import Path

import argparse
from scripts.drone_trait_extraction.drone_gis import process_tiff, find_drone_tiffs
from scripts.orthomosaic_generation import run_odm

# Define the Flask application for serving files
file_app = Flask(__name__)
latest_data = {'epoch': 0, 'map': 0}
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
    
@file_app.route('/list_dirs_nested', methods=['GET'])
def list_dirs_nested():
    global data_root_dir
    base_dir = Path(data_root_dir) / 'Raw'

    def build_nested_structure(path):
        structure = {}
        for child in path.iterdir():
            if child.is_dir():
                structure[child.name] = build_nested_structure(child)
        return structure

    nested_structure = build_nested_structure(base_dir)
    return jsonify(nested_structure), 200

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

    prefix = data_root_dir+'/Raw'
    image_folder = os.path.join(prefix, year, experiment, location, population, date, 'Drone', sensor, 'Images')

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
    processed_path = base_image_path.replace('/Raw/', 'Processed/').split('/Drone')[0] + '/Drone'
    save_directory = os.path.join(data_root_dir+'/', processed_path)
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

    processed_path = data['basePath'].replace('/Raw/', 'Processed/').split('/Drone')[0] + '/Drone'
    save_directory = os.path.join(data_root_dir+'/', processed_path)

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
    sensor = data.get('sensor')
    temp_dir = data.get('temp_dir')
    reconstruction_quality = data.get('reconstruction_quality')
    custom_options = data.get('custom_options')

    if not temp_dir:
        temp_dir = '/home/GEMINI/temp/project'
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
    args.sensor = sensor
    args.temp_dir = temp_dir
    args.reconstruction_quality = reconstruction_quality
    args.custom_options = custom_options
    
    # Run ODM in a separate thread
    thread = threading.Thread(target=run_odm, args=(args,))
    thread.start()

    return jsonify({"status": "success", "message": "ODM processing started successfully"})

### ROVER MODEL TRAINING ###
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
    global latest_data, training_stopped_event, new_folder
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
                while os.path.isfile(results_file):
                    try:
                        df = pd.read_csv(results_file, delimiter=',\s+', engine='python')
                        latest_data['epoch'] = int(df['epoch'].iloc[-1])  # Update latest epoch
                        latest_data['map'] = df['metrics/mAP50(B)'].iloc[-1]  # Update latest mAP
                    except Exception as e:
                        print(f"Error reading {results_file}: {e}")
                    time.sleep(5)  # Update every 30 seconds

        time.sleep(5)  # Check for new folders every 10 seconds
        
@file_app.route('/get_training_progress', methods=['GET'])
def get_training_progress():
    return jsonify(latest_data)

@file_app.route('/train_model', methods=['POST'])
def train_model():
    global data_root_dir, latest_data, training_stopped_event
    
    # receive the parameters
    epochs = int(request.json['epochs'])
    # epochs = 1 # testing
    batch_size = int(request.json['batchSize'])
    image_size = int(request.json['imageSize'])
    location = request.json['location']
    population = request.json['population']
    year = request.json['year']
    experiment = request.json['experiment']
    date = request.json['date']
    trait = request.json['trait']
    # sensor = request.json['sensor']
    sensor = 'Rover' # testing
    
    # extract labels
    labels_path = data_root_dir+'/Processed/'+year+'/'+experiment+'/'+location+'/'+population+'/'+date+'/'+sensor+'/Annotations/labels/train'
    labels = get_labels(labels_path)
    labels_arg = " ".join(labels)
    
    # other training args
    container_dir = '/app/mnt'
    pretrained = "/app/train/yolov8n.pt"
    save_train_model = container_dir+'/Processed/'+year+'/'+experiment+'/'+location+'/'+population+'/models/custom'
    scan_save = data_root_dir+'/Processed/'+year+'/'+experiment+'/'+location+'/'+population+'/models/custom'+f'/{trait}-det'
    latest_data['epoch'] = 0
    latest_data['map'] = 0
    training_stopped_event.clear()
    threading.Thread(target=scan_for_new_folders, args=(scan_save,), daemon=True).start()
    images = container_dir+'/Processed/'+year+'/'+experiment+'/'+location+'/'+population+'/'+date+'/'+sensor+'/Annotations'
    
    # run training
    cmd = (f"docker exec train "
           f"python /app/train/train.py "
           f"--pretrained {pretrained} --images {images} --save {save_train_model} --sensor {sensor} "
           f"--date {date} --trait {trait} --image-size {image_size} --epochs {epochs} "
           f"--batch-size {batch_size} --labels {labels_arg}")
    
    try:
        process = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = process.stdout.decode('utf-8')
        return jsonify({"message": "Training started", "output": output}), 202
    except subprocess.CalledProcessError as e:
        error_output = e.stderr.decode('utf-8')
        return jsonify({"error": error_output}), 500
    
@file_app.route('/stop_training', methods=['POST'])
def stop_training():
    global training_stopped_event, new_folder
    container_name = 'train'
    try:
        print('Training stopped by user.')
        kill_cmd = f"docker exec {container_name} pkill -9 -f python"
        subprocess.run(kill_cmd, shell=True)
        print(f"Sent SIGKILL to Python process in {container_name} container.")
        training_stopped_event.set()
        subprocess.run(f"rm -rf {new_folder}", check=True, shell=True)
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
