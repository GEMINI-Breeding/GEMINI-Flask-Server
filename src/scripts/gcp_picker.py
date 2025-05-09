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
import re

def clean_duplicates_in_msgs_synced(csv_path):
    """Remove duplicate entries from msgs_synced.csv based on image_path."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        original_count = len(df)
        df_cleaned = df.drop_duplicates(subset=['image_path'])
        if len(df_cleaned) < original_count:
            print(f"Removed {original_count - len(df_cleaned)} duplicate entries from {csv_path}")
            df_cleaned.to_csv(csv_path, index=False)
            return True
    return False

def calculate_distance(lat1, lon1, lat2, lon2):
    geod = Geod(ellps='WGS84')
    _, _, distance = geod.inv(lon1, lat1, lon2, lat2)
    return distance

def get_image_exif(image_path):
    # Extract GPS coordinates from EXIF data
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(e)
        return

    exif_data = image._getexif()
    latitude = None
    longitude = None
    altitude = None
    time_stamp = ""
    if exif_data is not None and 34853 in exif_data:
        # Get image dimensions
        width, height = image.size
        gps_info = exif_data[34853]
        latitude = float(gps_info[2][0] + gps_info[2][1] / 60 + gps_info[2][2] / 3600)
        if gps_info[1] == 'S':
            latitude = -latitude
        longitude = float(gps_info[4][0] + gps_info[4][1] / 60 + gps_info[4][2] / 3600)
        if gps_info[3] == 'W':
            longitude = -longitude
        altitude = float(gps_info[6])

    for tag_id in [36867, 36868, 306]:
        if tag_id in exif_data:
            # Get the valid time stamp
            time_stamp = exif_data[tag_id]
            break

    msg = {
            'image_path': image_path,
            'time': time_stamp,
            'lat': latitude,
            'lon': longitude,
            'alt':altitude,
            'naturalWidth': width,
            'naturalHeight': height
            }

    return msg

def natural_sort_key(s):
    """
    A key function for sorting strings containing numbers in a natural order
    Example: ['gcp1', 'gcp2', 'gcp10'] (correct) vs ['gcp1', 'gcp10', 'gcp2'] (incorrect)
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def collect_gcp_candidate(data_root_dir, image_folder, radius_meters):

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


    print("Loading gcp_locations.csv...")
    # Define the path to the predefined locations CSV file
    gcp_locations_csv = os.path.join(image_folder, '../../../../gcp_locations.csv')

    # Load predefined locations from CSV
    gcp_locations = []
    if not os.path.isfile(gcp_locations_csv):
        print("ERROR: Invalid selections: no gcp_locations.csv file found.")
    else:
        df = pd.read_csv(gcp_locations_csv)
        labels = df['Label'].tolist()
        latitudes = df['Lat_dec'].tolist()
        longitudes = df['Lon_dec'].tolist()
        if "Altitude" not in df:
            altitudes = np.zeros_like(longitudes)

        for i in range(len(labels)):
            gcp_locations.append({
                'label': labels[i],
                'latitude': latitudes[i],
                'longitude': longitudes[i],
                'altitude': altitudes[i]
            })

    msgs_synced_path = os.path.join(os.path.dirname(image_folder), "msgs_synced.csv")
    
    # Load msgs_synced.csv
    if os.path.isfile(msgs_synced_path):
        print("Loading msgs_synced.csv...")
        df = pd.read_csv(msgs_synced_path)
        image_names_in_df = df['image_path'].tolist()
    else:
        image_names_in_df = []

    selected_images = []
    for filename in tqdm(files):
        image_path = os.path.join(image_folder, filename)
        try:
            if image_path in image_names_in_df:
                index = image_names_in_df.index(image_path)
                # Use the DataFrame row at this index
                msg = df.iloc[index].to_dict()
            else:
                msg = get_image_exif(image_path)
                # If we got valid EXIF data, add it to the DataFrame and CSV
                if msg is not None:
                    # Add to our in-memory DataFrame
                    df = pd.concat([df, pd.DataFrame([msg])], ignore_index=True)
                    # Add to image_names_in_df list for subsequent lookups
                    image_names_in_df.append(filename)
        except (ValueError, IndexError) as e:
            print(f"Error finding index for {filename}: {e}")
            msg = get_image_exif(image_path)
            # If we got valid EXIF data from the fallback, save it too
            if msg is not None:
                df = pd.concat([df, pd.DataFrame([msg])], ignore_index=True)
                image_names_in_df.append(filename)

    
        if msg is not None:
            if len(gcp_locations) > 0 and (msg['lat'] is not None) and (msg['lon'] is not None):
                # Check if the image is within the predefined locations
                closest_dist = float('inf')
                closest_location = None
                for location in gcp_locations:
                    dist = calculate_distance(msg['lat'], msg['lon'], location['latitude'], location['longitude'])
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
                        'naturalWidth': msg['naturalWidth'],
                        'naturalHeight': msg['naturalWidth']
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
                    'naturalWidth': msg['naturalWidth'],
                    'naturalHeight': msg['naturalWidth']
                })


    # Sort the selected_images by gcp_label
    selected_images.sort(key=lambda x: natural_sort_key(x['gcp_label']))

    # Create a new CSV with headers
    df.to_csv(msgs_synced_path, mode='w', header=True, index=False)

    return selected_images


def refresh_gcp_candidate(data_root_dir, image_folder, radius_meters):

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


    print("Loading gcp_locations.csv...")
    # Define the path to the predefined locations CSV file
    gcp_locations_csv = os.path.join(image_folder, '../../../../gcp_locations.csv')

    # Load predefined locations from CSV
    gcp_locations = []
    if not os.path.isfile(gcp_locations_csv):
        print("ERROR: Invalid selections: no gcp_locations.csv file found.")
    else:
        df = pd.read_csv(gcp_locations_csv)
        labels = df['Label'].tolist()
        latitudes = df['Lat_dec'].tolist()
        longitudes = df['Lon_dec'].tolist()
        if "Altitude" not in df:
            altitudes = np.zeros_like(longitudes)

        for i in range(len(labels)):
            gcp_locations.append({
                'label': labels[i],
                'latitude': latitudes[i],
                'longitude': longitudes[i],
                'altitude': altitudes[i]
            })

    msgs_synced_path = os.path.join(os.path.dirname(image_folder), "msgs_synced.csv")
    
    # Load msgs_synced.csv
    if os.path.isfile(msgs_synced_path):
        print("Loading msgs_synced.csv...")
        df = pd.read_csv(msgs_synced_path)
        image_names_in_df = df['image_path'].tolist()
    else:
        image_names_in_df = []

    # Read gcp_list
    

    selected_images = []
    for filename in tqdm(files):
        image_path = os.path.join(image_folder, filename)
        try:
            if image_path in image_names_in_df:
                index = image_names_in_df.index(image_path)
                # Use the DataFrame row at this index
                msg = df.iloc[index].to_dict()
            else:
                msg = get_image_exif(image_path)
                # If we got valid EXIF data, add it to the DataFrame and CSV
                if msg is not None:
                    # Add to our in-memory DataFrame
                    df = pd.concat([df, pd.DataFrame([msg])], ignore_index=True)
                    # Add to image_names_in_df list for subsequent lookups
                    image_names_in_df.append(filename)
        except (ValueError, IndexError) as e:
            print(f"Error finding index for {filename}: {e}")
            msg = get_image_exif(image_path)
            # If we got valid EXIF data from the fallback, save it too
            if msg is not None:
                df = pd.concat([df, pd.DataFrame([msg])], ignore_index=True)
                image_names_in_df.append(filename)

    
        if msg is not None:
            if len(gcp_locations) > 0 and (msg['lat'] is not None) and (msg['lon'] is not None):
                # Check if the image is within the predefined locations
                closest_dist = float('inf')
                closest_location = None
                for location in gcp_locations:
                    dist = calculate_distance(msg['lat'], msg['lon'], location['latitude'], location['longitude'])
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
                        'naturalWidth': msg['naturalWidth'],
                        'naturalHeight': msg['naturalWidth']
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
                    'naturalWidth': msg['naturalWidth'],
                    'naturalHeight': msg['naturalWidth']
                })


    # Sort the selected_images by gcp_label
    selected_images.sort(key=lambda x: natural_sort_key(x['gcp_label']))

    # Create a new CSV with headers
    df.to_csv(msgs_synced_path, mode='w', header=True, index=False)

    return selected_images


