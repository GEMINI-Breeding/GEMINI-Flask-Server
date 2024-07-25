#### IMAGE PROCESSING ENDPOINTS ####
# Function to calculate the distance between two coordinates using pyproj
import os
import numpy as np
import pandas as pd
from pyproj import Geod
from tqdm import tqdm
from PIL import Image

import base64
import io

def calculate_distance(lat1, lon1, lat2, lon2):
    geod = Geod(ellps='WGS84')
    _, _, distance = geod.inv(lon1, lat1, lon2, lat2)
    return distance

def collect_gcp_candidate(data_root_dir, image_folder, radius_meters):
    global image_dict

    # Select the image folder
    if not os.path.isdir(image_folder):
        raise Exception("Invalid selections: no image folder.")

    # Process each image in the folder
    files = os.listdir(image_folder)
    
    file_filtered = []
    for filename in files:
        if 'mask' not in filename:
            file_ext = os.path.splitext(filename)[1]
            if file_ext.lower() in ['.jpg', '.jpeg', '.png']:
                file_filtered.append(filename)
    file_filtered.sort()
    files = file_filtered

    if len(files) == 0:
        raise Exception("Invalid selections: no files found in folder.")
    # Check if there are npy file that contains the image names

    selected_images = []
    print("Loading predefined locations from CSV file...")

    # Define the path to the predefined locations CSV file
    predefined_locations_csv = os.path.join(image_folder, '../../../../gcp_locations.csv')
    print(predefined_locations_csv)
    # Load predefined locations from CSV
    predefined_locations = []
    if not os.path.isfile(predefined_locations_csv):
        print("ERROR: Invalid selections: no gcp_locations.csv file found.")
    else:
        df = pd.read_csv(predefined_locations_csv)
        labels = df['Label'].tolist()
        latitudes = df['Lat_dec'].tolist()
        longitudes = df['Lon_dec'].tolist()
        for i in range(len(labels)):
            predefined_locations.append({
                'label': labels[i],
                'latitude': latitudes[i],
                'longitude': longitudes[i]
            })

    npy_path = os.path.join(image_folder, '../image_names_final.npy')
    if os.path.exists(npy_path):
        saved_dict = np.load(npy_path, allow_pickle=True).item()
        saved_files = saved_dict['files']
        # Check if the files are the same
        if len(saved_files) == len(files):
            # Then just use the saved files
            selected_images = saved_dict['selected_images']

    if selected_images == []:
        #image_dict = {}
        for filename in tqdm(files):
            # print("Processing image: " + filename)
            image_path = os.path.join(image_folder, filename)

            # Extract GPS coordinates from EXIF data
            image = Image.open(image_path)
            
            if 0:
                buf = io.BytesIO()
                image.save(buf, format='PNG')
                image_bytes = buf.getvalue()
                base64_encoded_result = base64.b64encode(image_bytes)
                image_dict[image_path] = base64_encoded_result
            
            # Get image dimensions
            width, height = image.size

            exif_data = image._getexif()
            if exif_data is not None and 34853 in exif_data:
                gps_info = exif_data[34853]
                latitude = gps_info[2][0] + gps_info[2][1] / 60 + gps_info[2][2] / 3600
                longitude = gps_info[4][0] + gps_info[4][1] / 60 + gps_info[4][2] / 3600
                latitude = float(latitude)
                longitude = float(longitude) * -1

                if len(predefined_locations) > 0:
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

                else:
                    # Logic for when gcp files are not provided
                    # Remove the first part of the image path
                    image_path = image_path.replace(data_root_dir, '')
                    selected_images.append({
                        'image_path': image_path,
                        'gcp_lat': 0,
                        'gcp_lon': 0,
                        'gcp_label': 'N/A',
                        'naturalWidth': width,
                        'naturalHeight': height
                    })

                # Save the selected images to a dict
                if len(selected_images) > 0:
                    npy_path = os.path.join(image_folder, '../image_names.npy')
                    np.save(npy_path, {'files': files, 'selected_images': selected_images})

    # Save the selected images to a dict
    if selected_images != []:
        if os.path.exists(npy_path):
            final_npy_path = os.path.join(image_folder, '../image_names_final.npy')
            os.rename(npy_path, final_npy_path)

    return selected_images