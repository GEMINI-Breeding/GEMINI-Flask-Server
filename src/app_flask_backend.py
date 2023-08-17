import os
import subprocess
import threading
import uvicorn

from flask import Flask, send_from_directory, jsonify, request
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.middleware.cors import CORSMiddleware

import csv
from PIL import Image
from pyproj import Geod
import pandas as pd

# Define the Flask application for serving files
file_app = Flask(__name__)

#### FILE SERVING ENDPOINTS ####
# endpoint to serve files
@file_app.route('/files/<path:filename>')
def serve_files(filename):
    return send_from_directory('/home/GEMINI/GEMINI-Data', filename)

# endpoint to list files
@file_app.route('/list_dirs/<path:dir_path>', methods=['GET'])
def list_dirs(dir_path):
    dir_path = os.path.join('/home/GEMINI/GEMINI-Data', dir_path)  # join with base directory path
    if os.path.exists(dir_path):
        dirs = next(os.walk(dir_path))[1]
        return jsonify(dirs), 200
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
    # receive the parameters
    location = request.json['location']
    population = request.json['population']
    date = request.json['date']
    radius_meters = request.json['radius_meters']

    prefix = '/home/GEMINI/GEMINI-Data/Raw'
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
                    image_path = image_path.replace('/home/GEMINI/GEMINI-Data', '')

                    selected_images.append({
                        'image_path': image_path,
                        'gcp_lat': closest_location['latitude'],
                        'gcp_lon': closest_location['longitude']
                    })

    # Return the selected images and their corresponding GPS coordinates
    return jsonify({'selected_images': selected_images}), 200

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

    # Start the Titiler server using the subprocess module
    titiler_command = "uvicorn titiler.application.main:app --reload --port 8091"
    titiler_process = subprocess.Popen(titiler_command, shell=True)

    # Start the Flask server
    uvicorn.run(app, host="0.0.0.0", port=5001)

    # Terminate the Titiler server when the Flask server is shut down
    titiler_process.terminate()
