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
import piexif # pip install piexif
from datetime import datetime, timezone

from utils import convert_to_unix_timestamp

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
    # You'll need to install piexif: pip install piexif
    
    # Extract GPS coordinates from EXIF data
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(e)
        return None

    # Get image dimensions
    width, height = image.size
    
    # Flag to track if we need to rotate the image
    do_rotation = False
    
    # Check if image is in portrait mode (height > width)
    if height > width:
        do_rotation = True
        print(f"Image is in portrait orientation, will rotate: {os.path.basename(image_path)}")
    
    # Read EXIF data using both PIL and piexif for different operations
    exif_data = image._getexif() or {}
    
    # Read full EXIF data with piexif for modification
    try:
        exif_dict = piexif.load(image_path)
    except Exception as e:
        print(f"Error reading EXIF data with piexif: {e}")
        exif_dict = {'0th': {}, 'Exif': {}, 'GPS': {}, '1st': {}, 'thumbnail': None}

    
    # # Update the time string format to include timezone
    # time_string = f"{time_string}{tz_string}"

    latitude = None
    longitude = None
    altitude = None
    gps_time_string = None
    exif_update = False

    time_string = None
    unix_time_stamp = None

    # First check if standard timestamps exist
    for tag_id in [36867, 36868, 306]:  # DateTimeOriginal, DateTimeDigitized, DateTime
        if tag_id in exif_data:
            if exif_data[tag_id]:  # Make sure it's not empty
                time_string = exif_data[tag_id]
                break
    
    # Check GPS EXIF data
    if exif_data is not None and 34853 in exif_data:
        gps_info = exif_data[34853]
        # Get the latitude and longitude
        try:
            latitude = float(gps_info[2][0] + gps_info[2][1] / 60 + gps_info[2][2] / 3600)
            if gps_info[1] == 'S':
                latitude = -latitude
            longitude = float(gps_info[4][0] + gps_info[4][1] / 60 + gps_info[4][2] / 3600)
            if gps_info[3] == 'W':
                longitude = -longitude
            altitude = float(gps_info[6])
        except (KeyError, IndexError, TypeError) as e:
            print(f"Error extracting GPS coordinates: {e}")
        
        # Extract GPS date and time if available
        gps_date = None
        gps_time = None
        
        # Get GPS date (tag 29)
        if 29 in gps_info:
            gps_date = gps_info[29]
        
        # Get GPS time (tag 7)
        if 7 in gps_info:
            try:
                hour = int(gps_info[7][0].numerator / gps_info[7][0].denominator)
                minute = int(gps_info[7][1].numerator / gps_info[7][1].denominator)
                # Get full precision for seconds including fractional part
                second = int(gps_info[7][2].numerator / gps_info[7][2].denominator)
                gps_time = f"{hour:02d}:{minute:02d}:{second:02d}"
            except (AttributeError, ZeroDivisionError) as e:
                print(f"Error parsing GPS time: {e}")
        
        # If we have both GPS date and time, create standard timestamp
        if gps_date and gps_time:
            gps_time_string = f"{gps_date} {gps_time}"
            try:
                # Create datetime object in UTC
                dt = datetime.strptime(gps_time_string, '%Y:%m:%d %H:%M:%S').replace(tzinfo=timezone.utc)
                # gps_time_string = dt.strftime('%Y:%m:%d %H:%M:%S.%f %z')  
                unix_time_stamp = dt.timestamp()  # Get UNIX timestamp
            except ValueError as e:
                print(f"Error converting GPS time to timestamp: {e}")
                unix_time_stamp = None
    # If unix_time_stamp exists update time_string
    if unix_time_stamp is not None:
        # Update standard EXIF tags with GPS timestamp
        if '0th' not in exif_dict:
            exif_dict['0th'] = {}
        if 'Exif' not in exif_dict:
            exif_dict['Exif'] = {}
        
        # Convert to bytes for piexif
        dt_local = datetime.fromtimestamp(unix_time_stamp) # Can we use GPS info to calculate the right time zone?
        time_string = dt_local.strftime('%Y:%m:%d %H:%M:%S.%f %z') # Include timezone in output
        
        # Update all the standard time fields
        exif_dict['0th'][piexif.ImageIFD.DateTime] = time_string  # 306
        exif_dict['Exif'][piexif.ExifIFD.DateTimeOriginal] = time_string  # 36867
        exif_dict['Exif'][piexif.ExifIFD.DateTimeDigitized] = time_string  # 36868
        
        # Flag that we need to save changes
        exif_update = True

    # If we need to update EXIF or rotate the image
    if exif_update or do_rotation:
        try:
            if do_rotation:
                # Rotate the image 90 degrees clockwise
                rotated_image = image.transpose(Image.ROTATE_270)
                # Update dimensions after rotation
                width, height = rotated_image.size
                # Use the rotated image for saving
                image = rotated_image
            
            # Convert EXIF data to bytes
            exif_bytes = piexif.dump(exif_dict)
            
            # Save the image with updated EXIF and/or rotation
            image.save(image_path, exif=exif_bytes)
            
            if exif_update:
                print(f"Updated EXIF timestamps in {os.path.basename(image_path)}")
            if do_rotation:
                print(f"Rotated {os.path.basename(image_path)} to landscape orientation")
        except Exception as e:
            print(f"Error updating image: {e}")
    
    # Create and return the message dictionary with potentially updated dimensions
    msg = {
        'image_path': image_path,
        'time': time_string,
        'timestamp': unix_time_stamp,
        'lat': latitude,
        'lon': longitude,
        'alt': altitude,
        'naturalWidth': width,
        'naturalHeight': height
    }

    return msg


def process_exif_data_async(file_paths, data_type, msgs_synced_file, existing_df, existing_paths):
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
            
    # check if image has any exif data
    if not exif_data_list:
        print("No new EXIF data extracted. You may need to upload a msgs_synced.csv (synced metadata) file manually.")

def natural_sort_key(s):
    """
    A key function for sorting strings containing numbers in a natural order
    Example: ['gcp1', 'gcp2', 'gcp10'] (correct) vs ['gcp1', 'gcp10', 'gcp2'] (incorrect)
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def image_selection(data_root_dir, image_folder, files, df_msgs_synced,
                    image_names_in_df,gcp_locations,radius_meters, check_exif=True):
    
    selected_images = []
    len_files = len(files)
    print(f"Selecting images out of {len_files} files in {image_folder}...")
    for filename in tqdm(files):
        image_path = os.path.join(image_folder, filename)
        normalized_filename = f'/top/{filename}'
        
        # try:
        if image_path in image_names_in_df:
            # print('Using image_path')
            index = image_names_in_df.index(image_path)
            # Use the DataFrame row at this index
            msg = df_msgs_synced.iloc[index].to_dict()
        elif normalized_filename in image_names_in_df:
            # print('Using normalized filename')
            index = image_names_in_df.index(normalized_filename)
            msg = df_msgs_synced.iloc[index].to_dict()
            
            # get naturalWidth and naturalHeight from the file
            with Image.open(image_path) as img:
                msg['naturalWidth'] = img.width
                msg['naturalHeight'] = img.height
            
        elif check_exif:
            msg = get_image_exif(image_path)
            # If we got valid EXIF data, add it to the DataFrame and CSV
            if msg is not None:
                # Add to our in-memory DataFrame
                df_msgs_synced = pd.concat([df_msgs_synced, pd.DataFrame([msg])], ignore_index=True)
                # Add to image_names_in_df list for subsequent lookups
                image_names_in_df.append(filename)             
        else:
            print(f"Skipping {filename}: no EXIF data found")
            continue
        # except (ValueError, IndexError) as e:
        #     print(f"Error finding index for {filename}: {e}")
        #     msg = get_image_exif(image_path)
        #     # If we got valid EXIF data from the fallback, save it too
        #     if msg is not None:
        #         df_msgs_synced = pd.concat([df_msgs_synced, pd.DataFrame([msg])], ignore_index=True)
        #         image_names_in_df.append(filename)
        #     else:
        #         print(f"Skipping {filename}: no EXIF data found")
        #         continue

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

def write_geo_txt(df_msgs_synced, geo_txt_path, srs="EPSG:4326"):
    """
    Write a geo.txt file for ODM from df_msgs_synced, skipping images without GPS info.
    """
    with open(geo_txt_path, 'w') as f:
        f.write(f"{srs}\n")
        for _, row in df_msgs_synced.iterrows():
            # Only write if both lat and lon are present and not NaN
            if pd.notna(row.get('lat')) and pd.notna(row.get('lon')):
                # if 'image_path' is not in row, use '/top/rgb_file'
                if 'image_path' in row:
                    image_name = os.path.basename(row['image_path'])
                else:
                    image_name = os.path.basename(row['/top/rgb_file'])
                lon = row['lon']
                lat = row['lat']
                alt = row['alt'] if pd.notna(row.get('alt')) else 0
                # yaw, pitch, roll, h_accuracy, v_accuracy = 0 if unknown
                f.write(f"{image_name} {lon} {lat} {alt} 0 0 0 0 0\n")

def collect_gcp_candidate(data_root_dir, image_folder, radius_meters):
    

    # Select the image folder
    if not os.path.isdir(image_folder):
        raise Exception(f"Invalid selections: no image folder {image_folder}.")

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
        raise Exception(f"Invalid selections: no files found in folder {image_folder}.")


    print("Loading gcp_locations.csv...")
    # Define the path to the predefined locations CSV file
    gcp_locations_csv = os.path.join(image_folder, '../../../../gcp_locations.csv')

    # Load predefined locations from CSV
    gcp_locations = []
    if not os.path.isfile(gcp_locations_csv):
        print("WARNING: Invalid selections: no gcp_locations.csv file found.")
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

    possible_msgs_synced_paths = [
        os.path.join(os.path.dirname(image_folder), "msgs_synced.csv"),
        os.path.join(os.path.dirname(
            os.path.dirname(image_folder)
        ), "Metadata", "msgs_synced.csv"),
    ]
    for possible_msgs_synced_path in possible_msgs_synced_paths:
        if os.path.isfile(possible_msgs_synced_path):
            print(f"Found msgs_synced.csv at {possible_msgs_synced_path}")
            msgs_synced_path = possible_msgs_synced_path
            # msgs_synced_path = os.path.join(
            #     os.path.dirname(
            #         os.path.dirname(image_folder)
            #     ), "Metadata", "msgs_synced.csv"
            # )
            break
        else:
            print(f"msgs_synced.csv not found at {possible_msgs_synced_path}.")
            msgs_synced_path = None
    
    # Load msgs_synced.csv
    if os.path.isfile(msgs_synced_path):
        df_msgs_synced = pd.read_csv(msgs_synced_path)
        # print(df_msgs_synced.columns)
        if 'image_path' in df_msgs_synced.columns:
            print('checking image_path')
            check_exif = True
            image_names_in_df = df_msgs_synced['image_path'].tolist()
        else:
            print('checking /top/rgb_file')
            check_exif = False
            image_names_in_df = df_msgs_synced['/top/rgb_file'].tolist()
    else:
        print(f"msgs_synced.csv not found at {msgs_synced_path}")
        image_names_in_df = []
    print(f'Length of image_names_in_df: {len(image_names_in_df)}')

    selected_images, df_msgs_synced, image_names_in_df = image_selection(data_root_dir,image_folder,
                                                                         files, df_msgs_synced,
                                                                         image_names_in_df,gcp_locations,
                                                                         radius_meters, check_exif=check_exif)
    if 0:
        # Sort the selected_images by gcp_label
        selected_images.sort(key=lambda x: natural_sort_key(x['gcp_label']))

    # Create a new CSV with headers
    df_msgs_synced.to_csv(msgs_synced_path, mode='w', header=True, index=False)

    # Check if drone_msgs.csv exists
    drone_msgs_path = os.path.join(os.path.dirname(image_folder), "drone_msgs.csv")
    if os.path.exists(drone_msgs_path):
        print(f"Found drone_msgs.csv at {drone_msgs_path}")
        # Update df_msgs_synced lat, lon, and alt using drone_msgs.csv
        df_msgs_synced = update_msgs_synced_with_drone_data(df_msgs_synced, drone_msgs_path)
        
        # Save updated data back to CSV
        df_msgs_synced.to_csv(msgs_synced_path, mode='w', header=True, index=False)
        print("Updated msgs_synced.csv with drone GPS data")
        
    # Create a geo.txt for ODM
    print("Create a geo.txt for ODM")
    geo_txt_path = os.path.join(os.path.dirname(image_folder), "geo.txt")
    write_geo_txt(df_msgs_synced, geo_txt_path)

    return selected_images


def refresh_gcp_candidate(data_root_dir, image_folder, radius_meters):

    # Select the image folder
    if not os.path.isdir(image_folder):
        raise Exception(f"Invalid selections: no image folder {image_folder}.")

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
        raise Exception(f"Invalid selections: no files found in folder {image_folder}.")


    print("Loading gcp_locations.csv...")
    # Define the path to the predefined locations CSV file
    gcp_locations_csv = os.path.join(image_folder, '../../../../gcp_locations.csv')

    # Load predefined locations from CSV
    gcp_locations = []
    if not os.path.isfile(gcp_locations_csv):
        print("WARNING: Invalid selections: no gcp_locations.csv file found.")
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

    possible_msgs_synced_paths = [
        os.path.join(os.path.dirname(image_folder), "msgs_synced.csv"),
        os.path.join(os.path.dirname(
            os.path.dirname(image_folder)
        ), "Metadata", "msgs_synced.csv"),
    ]
    for msgs_synced_path in possible_msgs_synced_paths:
        if os.path.isfile(msgs_synced_path):
            print(f"Found msgs_synced.csv at {msgs_synced_path}")
            # msgs_synced_path = os.path.join(
            #     os.path.dirname(
            #         os.path.dirname(image_folder)
            #     ), "Metadata", "msgs_synced.csv"
            # )
            break
        else:
            print(f"msgs_synced.csv not found at {msgs_synced_path}.")
            msgs_synced_path = None
    
    # Load msgs_synced.csv
    if os.path.isfile(msgs_synced_path):
        df_msgs_synced = pd.read_csv(msgs_synced_path)
        if 'image_path' in df_msgs_synced.columns:
            print('checking image_path')
            image_names_in_df = df_msgs_synced['image_path'].tolist()
        else:
            print('checking /top/rgb_file')
            check_exif = False
            image_names_in_df = df_msgs_synced['/top/rgb_file'].tolist()
    else:
        print(f"msgs_synced.csv not found at {msgs_synced_path}")
        image_names_in_df = []
    print(f'Length of image_names_in_df: {len(image_names_in_df)}')
    image_names_in_df.sort()

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
                                                                         radius_meters, check_exif=check_exif)
    # Apply projection matrix correction to GPS coordinates
    print("Updating GPS coordinates based on GCP points...")
    df_msgs_synced = update_gps_with_projection_matrix(df_msgs_synced, gcp_list_image_names, df_lat, df_lon)
    # df_msgs_synced.to_csv(msgs_synced_path, index=False)    

    new_selected_images, df_msgs_synced, image_names_in_df = image_selection(data_root_dir,image_folder,
                                                                        files, df_msgs_synced,
                                                                        image_names_in_df,gcp_locations,
                                                                        radius_meters, check_exif=check_exif)
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


def gcp_picker_save_array(data_root_dir, data, debug=False):
    # Extracting the directory path based on the first element in the array 
    base_image_path = data['array'][0]['image_path']
    platform = data['platform']
    sensor = data['sensor']
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
            if debug:
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

    return filename

def update_msgs_synced_with_drone_data(df_msgs_synced, drone_msgs_path):
    """
    Update df_msgs_synced lat, lon, and alt using drone_msgs.csv with KD tree timestamp matching
    """
    from sklearn.neighbors import NearestNeighbors
    import pandas as pd
    import numpy as np
    
    print(f"Loading drone messages from {drone_msgs_path}")
    df_drone = pd.read_csv(drone_msgs_path)
    
    # Convert timestamps for both datasets
    print("Converting timestamps for matching...")
    
    # For msgs_synced
    if 'timestamp' in df_drone.columns:
        df_msgs_synced['unix_time'] = df_msgs_synced['timestamp']
    else:
        df_msgs_synced['unix_time'] = df_msgs_synced['time'].apply(convert_to_unix_timestamp)
    valid_msgs_mask = df_msgs_synced['unix_time'].notna()
    valid_msgs_count = valid_msgs_mask.sum()
    
    # For drone_msgs
    if 'timestamp' in df_drone.columns:
        df_drone['unix_time'] = df_drone['timestamp']
    elif 'time' in df_drone.columns:
        df_drone['unix_time'] = df_drone['time'].apply(convert_to_unix_timestamp)
    else:
        print("ERROR: No timestamp column found in drone_msgs.csv")
        return df_msgs_synced
    
    valid_drone_mask = df_drone['unix_time'].notna()
    valid_drone_count = valid_drone_mask.sum()
    
    print(f"Valid timestamps: {valid_msgs_count} in msgs_synced, {valid_drone_count} in drone_msgs")
    
    if valid_msgs_count == 0 or valid_drone_count == 0:
        print("No valid timestamps found for matching")
        return df_msgs_synced
    
    # Filter to valid timestamps only
    df_msgs_valid = df_msgs_synced[valid_msgs_mask].copy()
    df_drone_valid = df_drone[valid_drone_mask].copy()
    
    # Prepare data for KD tree (timestamps as 1D array)
    drone_timestamps = df_drone_valid['unix_time'].values.reshape(-1, 1)
    msgs_timestamps = df_msgs_valid['unix_time'].values.reshape(-1, 1)
    
    # Create KD tree for drone timestamps
    print("Building KD tree for timestamp matching...")
    kd_tree = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
    kd_tree.fit(drone_timestamps)
    
    # Find nearest drone timestamp for each image timestamp
    distances, indices = kd_tree.kneighbors(msgs_timestamps)
    
    # Set maximum time difference threshold (e.g., 5 seconds)
    max_time_diff = 5.0  # seconds
    
    updates_count = 0
    for i, (distance, drone_idx) in enumerate(zip(distances.flatten(), indices.flatten())):
        if distance <= max_time_diff:
            # Get the corresponding rows
            msgs_idx = df_msgs_valid.index[i]
            drone_row = df_drone_valid.iloc[drone_idx]
            
            # Update coordinates if available in drone data
            if 'lat' in drone_row and pd.notna(drone_row['lat']):
                df_msgs_synced.at[msgs_idx, 'lat'] = drone_row['lat']
            if 'lon' in drone_row and pd.notna(drone_row['lon']):
                df_msgs_synced.at[msgs_idx, 'lon'] = drone_row['lon']
            if 'alt' in drone_row and pd.notna(drone_row['alt']):
                df_msgs_synced.at[msgs_idx, 'alt'] = drone_row['alt']
            elif 'altitude' in drone_row and pd.notna(drone_row['altitude']):
                df_msgs_synced.at[msgs_idx, 'alt'] = drone_row['altitude']
            
            updates_count += 1
        else:
            print(f"Timestamp difference too large: {distance:.2f}s for image at index {i}")
    
    print(f"Updated {updates_count} entries with drone GPS data")
    
    # Clean up temporary column
    df_msgs_synced.drop('unix_time', axis=1, inplace=True)
    
    return df_msgs_synced