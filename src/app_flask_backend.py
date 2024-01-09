import os
import subprocess
import threading
import uvicorn

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

import argparse
from scripts.drone_trait_extraction.drone_gis import process_tiff, find_drone_tiffs


# Define the Flask application for serving files
file_app = Flask(__name__)

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

    prefix = data_root_dir+'/Raw'
    image_folder = os.path.join(prefix, location, population, date, 'Drone', 'Images')

    print("Loading predefined locations from CSV file...")

    # Define the path to the predefined locations CSV file
    predefined_locations_csv = os.path.join(prefix, location, population, 'gcp_locations.csv')

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


@file_app.route('/process_drone_tiff/<path:dir_path>')
def process_drone_tiff(dir_path):
    # Check if already in processing
    global now_drone_processing
    if now_drone_processing:
        return jsonify({'message': 'Already in processing'}), 400
    
    now_drone_processing = True
    print(f"Processing drone tiff...{dir_path}")
    # Define the path to the image folder
    image_folder = os.path.join(data_root_dir, "Processed", dir_path,"Drone")
    
    try: 
        rgb_tif_file, dem_tif_file, thermal_tif_file = find_drone_tiffs(image_folder)
        geojson_path = os.path.join(data_root_dir, "Processed", dir_path,"../Plot-Attributes-WGS84.geojson")
        date = dir_path.split("/")[-1]
        sensor = "Drone"
        output_geojson = os.path.join(data_root_dir, "Processed", dir_path,"Results",f"{date}-{sensor}-Traits-WGS84.geojson")
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
            for line in lines:
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
            for line in lines:
                parts = line.strip().split()
                existing_data.append({
                    'gcp_lon': parts[0],
                    'gcp_lat': parts[1],
                    'pointX': parts[3],
                    'pointY': parts[4],
                    'image_path': os.path.join(processed_path, parts[5])  # Assuming this path is correct, adjust if needed
                })
    else:
        # Create the file if it doesn't exist
        with open(filename, 'w') as f:
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

    prefix = data_root_dir+'/Processed'
    traitpth = os.path.join(prefix, location, population, date, 'Results', 
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
    csv_data = data.get('csvData')
    filename = data.get('filename')

    # Construct file path based on GCP variables
    prefix = data_root_dir+'/Processed'
    file_path = os.path.join(prefix, selected_location_gcp, selected_population_gcp, filename)

    # Save CSV data to file
    with open(file_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)

    return jsonify({"status": "success", "message": "CSV data saved successfully"}), 200
    
@file_app.route('/save_geojson', methods=['POST'])
def save_geojson():
    data = request.json
    selected_location_gcp = data.get('selectedLocationGcp')
    selected_population_gcp = data.get('selectedPopulationGcp')
    geojson_data = data.get('geojsonData')
    filename = data.get('filename')

    # Load GeoJSON data into a GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(geojson_data)

    # Construct file path based on GCP variables
    prefix = data_root_dir+'/Processed'
    file_path = os.path.join(prefix, selected_location_gcp, selected_population_gcp, filename)

    # Save GeoDataFrame to file
    gdf.to_file(file_path, driver='GeoJSON')

    return jsonify({"status": "success", "message": "GeoJSON data saved successfully"})

@file_app.route('/load_geojson', methods=['GET'])
def load_geojson():
    selected_location_gcp = request.args.get('selectedLocationGcp')
    selected_population_gcp = request.args.get('selectedPopulationGcp')
    filename = request.args.get('filename')

    # Construct file path
    prefix = data_root_dir+'/Processed'
    file_path = os.path.join(prefix, selected_location_gcp, selected_population_gcp, filename)

    # Load GeoJSON data from file
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            geojson_data = json.load(file)
        return jsonify(geojson_data)
    else:
        return jsonify({"status": "error", "message": "File not found"})

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
    parser.add_argument('--data_root_dir', type=str, default='/home/GEMINI/GEMINI-Data',required=False)
    parser.add_argument('--port', type=int, default=5000,required=False) # Default port is 5000
    args = parser.parse_args()

    # Print the arguments to the console
    print(f"data_root_dir: {args.data_root_dir}")
    print(f"port: {args.port}")

    # Update global data_root_dir from the argument
    global data_root_dir
    data_root_dir = args.data_root_dir

    global now_drone_processing
    now_drone_processing = False

    # Start the Titiler server using the subprocess module
    titiler_command = "uvicorn titiler.application.main:app --reload --port 8090"
    titiler_process = subprocess.Popen(titiler_command, shell=True)

    # Start the Flask server
    uvicorn.run(app, host="127.0.0.1", port=args.port)

    # Terminate the Titiler server when the Flask server is shut down
    titiler_process.terminate()
