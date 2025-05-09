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

def image_selection(data_root_dir, image_folder, files, df_msgs_synced,
                    image_names_in_df,gcp_locations,radius_meters):
    
    selected_images = []
    for filename in tqdm(files):
        image_path = os.path.join(image_folder, filename)
        try:
            if image_path in image_names_in_df:
                index = image_names_in_df.index(image_path)
                # Use the DataFrame row at this index
                msg = df_msgs_synced.iloc[index].to_dict()
            else:
                msg = get_image_exif(image_path)
                # If we got valid EXIF data, add it to the DataFrame and CSV
                if msg is not None:
                    # Add to our in-memory DataFrame
                    df_msgs_synced = pd.concat([df_msgs_synced, pd.DataFrame([msg])], ignore_index=True)
                    # Add to image_names_in_df list for subsequent lookups
                    image_names_in_df.append(filename)
        except (ValueError, IndexError) as e:
            print(f"Error finding index for {filename}: {e}")
            msg = get_image_exif(image_path)
            # If we got valid EXIF data from the fallback, save it too
            if msg is not None:
                df_msgs_synced = pd.concat([df_msgs_synced, pd.DataFrame([msg])], ignore_index=True)
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
                        'naturalHeight': msg['naturalHeight']
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
                    'naturalHeight': msg['naturalHeight']
                })
    return selected_images, df_msgs_synced, image_names_in_df

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
        df_gcplocations = pd.read_csv(gcp_locations_csv)
        labels = df_gcplocations['Label'].tolist()
        latitudes = df_gcplocations['Lat_dec'].tolist()
        longitudes = df_gcplocations['Lon_dec'].tolist()
        if "Altitude" not in df_gcplocations:
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
        df_msgs_synced = pd.read_csv(msgs_synced_path)
        image_names_in_df = df_msgs_synced['image_path'].tolist()
    else:
        image_names_in_df = []

    selected_images, df_msgs_synced, image_names_in_df = image_selection(data_root_dir,image_folder,
                                                                         files, df_msgs_synced,
                                                                         image_names_in_df,gcp_locations,
                                                                         radius_meters)
    if 0:
        # Sort the selected_images by gcp_label
        selected_images.sort(key=lambda x: natural_sort_key(x['gcp_label']))

    # Create a new CSV with headers
    df_msgs_synced.to_csv(msgs_synced_path, mode='w', header=True, index=False)

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
        df_gcplocations = pd.read_csv(gcp_locations_csv)
        labels = df_gcplocations['Label'].tolist()
        latitudes = df_gcplocations['Lat_dec'].tolist()
        longitudes = df_gcplocations['Lon_dec'].tolist()
        if "Altitude" not in df_gcplocations:
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
        df_msgs_synced = pd.read_csv(msgs_synced_path)
        image_names_in_df = df_msgs_synced['image_path'].tolist()
    else:
        image_names_in_df = []

    # Read gcp_list
    gcp_list_path = os.path.join(os.path.dirname(image_folder.replace('Raw','Intermediate')),"gcp_list.txt")
    with open(gcp_list_path, 'r') as f:
        epsg_code = f.readline().strip()  # "EPSG:4326" 
        print(f"CRS: {epsg_code}")

    df_gcp_list = pd.read_csv(
        gcp_list_path,
        header=None,
        skiprows=1, 
        sep=' ',
    )
    df_lon = df_gcp_list[0]
    df_lat = df_gcp_list[1]
    gcp_list_image_names = df_gcp_list[5]

    # Baseline
    selected_images, df_msgs_synced, image_names_in_df = image_selection(data_root_dir,image_folder,
                                                                         files, df_msgs_synced,
                                                                         image_names_in_df,gcp_locations,
                                                                         radius_meters)
    # Apply projection matrix correction to GPS coordinates
    print("Updating GPS coordinates based on GCP points...")
    df_msgs_synced = update_gps_with_projection_matrix(df_msgs_synced, gcp_list_image_names, df_lat, df_lon)
    # df_msgs_synced.to_csv(msgs_synced_path, index=False)    

    new_selected_images, df_msgs_synced, image_names_in_df = image_selection(data_root_dir,image_folder,
                                                                        files, df_msgs_synced,
                                                                        image_names_in_df,gcp_locations,
                                                                        radius_meters)
    # Compare and merge selected_images with new_selected_images
    # Create a dictionary of image paths from the original selected_images for quick lookup
    original_image_paths = {img['image_path']: True for img in selected_images}
    
    # Create a merged list with original items first
    merged_images = selected_images.copy()
    
    # Append only new items from new_selected_images
    new_items_count = 0
    for img in new_selected_images:
        if img['image_path'] not in original_image_paths:
            merged_images.append(img)
            new_items_count += 1
    
    if new_items_count > 0:
        print(f"Added {new_items_count} new images to the selection")

    if 0:
        # Sort the selected_images by gcp_label
        selected_images.sort(key=lambda x: natural_sort_key(x['gcp_label']))

    # Create a new CSV with headers
    df_msgs_synced.to_csv(msgs_synced_path, mode='w', header=True, index=False)

    return merged_images

def update_gps_with_projection_matrix(df_msgs_synced, gcp_list_image_names, df_lat, df_lon):
    """
    Update GPS coordinates in msgs_synced.csv using projection matrix calculated from GCP points
    
    Steps:
    1. Match images between gcp_list and msgs_synced.csv
    2. Extract lat/lon pairs from both sources
    3. Calculate transformation matrix
    4. Apply transformation to all coordinates in msgs_synced.csv
    """
    import numpy as np

    # Extract base filenames for matching
    df_msgs_synced['base_filename'] = df_msgs_synced['image_path'].apply(lambda x: os.path.basename(x))
    
    # Create lists to store matching coordinates
    source_coords = []  # Coordinates from msgs_synced.csv
    target_coords = []  # Coordinates from GCP list
    
    # Find matching images and collect coordinate pairs
    matches_count = 0
    for i, gcp_img_name in enumerate(gcp_list_image_names):
        matching_rows = df_msgs_synced[df_msgs_synced['base_filename'] == gcp_img_name]
        if not matching_rows.empty:
            for _, row in matching_rows.iterrows():
                if pd.notna(row['lat']) and pd.notna(row['lon']):
                    source_coords.append([row['lon'], row['lat'], 1])  # Homogeneous coordinates [lon, lat, 1]
                    target_coords.append([df_lon[i], df_lat[i]])       # [lon, lat]
                    matches_count += 1
    
    print(f"Found {matches_count} matching image coordinates")
    
    # Convert to numpy arrays
    source_coords = np.array(source_coords)
    target_coords = np.array(target_coords)
    
    if len(source_coords) < 3:
        print("Not enough matching points to calculate transformation (need at least 3)")
        return False
    
    # Calculate affine transformation matrix using least squares
    # For transformation [x', y'] = A [x, y, 1]
    # where A is a 2x3 matrix
    A, residuals, rank, s = np.linalg.lstsq(source_coords, target_coords, rcond=None)
    
    # A is transposed, so we get a 3x2 matrix. We need to transpose it back
    A = A.T
    
    print(f"Transformation matrix calculated:\n{A}")
    
    # Apply transformation to all valid coordinates in df_msgs
    valid_indices = df_msgs_synced[pd.notna(df_msgs_synced['lat']) & pd.notna(df_msgs_synced['lon'])].index
    
    if len(valid_indices) > 0:
        # Extract coordinates and add homogeneous component
        coords = df_msgs_synced.loc[valid_indices, ['lon', 'lat']].values
        homog_coords = np.hstack((coords, np.ones((len(coords), 1))))
        
        # Apply transformation
        new_coords = np.dot(homog_coords, A.T)
        
        # Update DataFrame
        df_msgs_synced.loc[valid_indices, 'lon'] = new_coords[:, 0]
        df_msgs_synced.loc[valid_indices, 'lat'] = new_coords[:, 1]
    
    # Remove temporary column and save updated file
    df_msgs_synced.drop('base_filename', axis=1, inplace=True)
    print(f"Updated {len(valid_indices)} coordinates")

    return df_msgs_synced


