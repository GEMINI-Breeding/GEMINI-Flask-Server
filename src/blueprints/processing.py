# Standard library imports
import os
import subprocess
import threading
import shutil
import json
import pandas as pd
import geopandas as gpd
import numpy as np
import traceback
import sys
from flask import Blueprint, make_response, jsonify, request, send_file, current_app
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


# Local application/library specific imports
from scripts.drone_trait_extraction import shared_states
from scripts.drone_trait_extraction.drone_gis import process_tiff, find_drone_tiffs, query_drone_images
from scripts.orthomosaic_generation import run_odm,reset_odm, make_odm_args, monitor_log_updates
from scripts.gcp_picker import collect_gcp_candidate, refresh_gcp_candidate
from scripts.stitch_utils import (
    run_stitch_all_plots,
    create_combined_mosaic_separate
)
# stitch pipeline
import sys
AGROWSTITCH_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../AgRowStitch"))
print(AGROWSTITCH_PATH)
sys.path.append(AGROWSTITCH_PATH)
from scripts.stitch_utils import (
    monitor_stitch_updates_multi_plot
)

processing_bp = Blueprint('processing', __name__)

def complete_stitch_workflow(msgs_synced_path, image_path, config_path, custom_options, 
                            save_path, image_calibration, stitch_stop_event, progress_callback, monitoring_stop_event=None):
    """
    Complete workflow function that handles both stitching and mosaic creation
    """
    try:
        # Run the main stitching process
        print("=== STARTING MAIN STITCHING PROCESS ===")
        stitch_results = run_stitch_all_plots(
            msgs_synced_path, image_path, config_path, custom_options, 
            save_path, image_calibration, stitch_stop_event, progress_callback
        )
        
        # Check if stitching was successful and we have results
        if stitch_results and stitch_results.get('processed_plots'):
            print("=== MAIN STITCHING COMPLETED, STARTING MOSAIC CREATION ===")
            
            # Create the combined mosaic separately to avoid multiprocessing issues
            mosaic_success = create_combined_mosaic_separate(
                stitch_results['versioned_output_path'],
                stitch_results['processed_plots'],
                progress_callback,
                monitoring_stop_event  # Pass the monitoring stop event to cancel monitoring
            )
            
            if mosaic_success:
                print("=== COMPLETE WORKFLOW FINISHED SUCCESSFULLY ===")
            else:
                print("=== WORKFLOW COMPLETED WITH MOSAIC WARNINGS ===")
        else:
            print("=== WORKFLOW COMPLETED WITH ERRORS ===")
                
    except Exception as e:
        print(f"ERROR in complete_stitch_workflow: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
    finally:
        print("=== WORKFLOW FINALLY BLOCK - ENSURING COMPLETION ===")
        
        # Set final 100% progress - monitoring thread is already stopped by mosaic creation
        if progress_callback:
            print("Setting final progress to 100%...")
            progress_callback(100)
            print("Final progress set to 100% - workflow complete!")
        
        print("=== WORKFLOW FULLY COMPLETED ===")


#### SCRIPT SERVING ENDPOINTS ####
# endpoint to run script
@processing_bp.route('/run_script', methods=['POST'])
def run_script():
    data = request.json
    script_path = data.get('script_path')

    def run_in_thread(script_path):
        subprocess.call(script_path, shell=True)

    thread = threading.Thread(target=run_in_thread, args=(script_path,))
    thread.start()

    return jsonify({'message': 'Script started'}), 200


@processing_bp.route('/get_gcp_selcted_images', methods=['POST'])
def get_gcp_selcted_images():
    data_root_dir = current_app.config['DATA_ROOT_DIR']
    try:
        image_folder = os.path.join(data_root_dir, 'Raw', request.json['year'], 
                                    request.json['experiment'], request.json['location'], 
                                    request.json['population'], request.json['date'], 
                                    request.json['platform'], request.json['sensor'], 'Images')
        
        # if folder 'top' is in image_folder, add it to the path
        if os.path.isdir(os.path.join(image_folder, 'top')):
            print("Found 'top' folder in image_folder, adding it to the path")
            image_folder = os.path.join(image_folder, 'top')
        selected_images = collect_gcp_candidate(data_root_dir, image_folder, request.json['radius_meters'])
        status = "DONE"

        # Return the selected images and their corresponding GPS coordinates
        return jsonify({'selected_images': selected_images,
                        'num_total': len(selected_images),
                        'status':status}), 200
    
    except Exception as e:
        print(e)
        selected_images = []
        status = "DONE"
        return jsonify({'selected_images': selected_images,
                    'num_total': len(selected_images),
                    'status':status}), 200


@processing_bp.route('/refresh_gcp_selcted_images', methods=['POST'])
def refresh_gcp_selcted_images():
    data_root_dir = current_app.config['DATA_ROOT_DIR']
    try:
        image_folder = os.path.join(data_root_dir, 'Raw', request.json['year'], 
                                    request.json['experiment'], request.json['location'], 
                                    request.json['population'], request.json['date'], 
                                    request.json['platform'], request.json['sensor'], 'Images')
        # if folder 'top' is in image_folder, add it to the path
        if os.path.isdir(os.path.join(image_folder, 'top')):
            print("Found 'top' folder in image_folder, adding it to the path")
            image_folder = os.path.join(image_folder, 'top')
        selected_images = refresh_gcp_candidate(data_root_dir, image_folder, request.json['radius_meters'])
        status = "DONE"

        # Return the selected images and their corresponding GPS coordinates
        return jsonify({'selected_images': selected_images,
                        'num_total': len(selected_images),
                        'status':status}), 200
    except Exception as e:
        print(e)
        selected_images = []
        status = "DONE"
        return jsonify({'selected_images': selected_images,
                    'num_total': len(selected_images),
                    'status':status}), 200

@processing_bp.route('/get_drone_extract_progress', methods=['GET'])
def get_drone_extract_progress():
    global processed_image_folder
    latest_data = current_app.config['LATEST_DATA']
    # data = request.json
    # tiff_rgb = data['tiff_rgb']
    # print("Processed image folder: "+ processed_image_folder)
    txt_file = os.path.join(processed_image_folder, 'progress.txt')
    
    # Check if the file exists
    if os.path.exists(txt_file):
        with open(txt_file, 'r') as file:
            number = file.read().strip()
            if number == '':
                latest_data['drone_extract'] = 0
            else:
                latest_data['drone_extract'] = float(number)
        return jsonify(latest_data)
    else:
        return jsonify({'error': 'Drone progress not found'}), 404
    
@processing_bp.route('/stop_drone_extract', methods=['POST'])
def stop_drone_extract():
    latest_data = current_app.config['LATEST_DATA']
    try:
        shared_states.stop_signal = True
        print(f'Shared states variable changed: {shared_states.stop_signal}')
        latest_data['drone_extract'] = 0
        print('Drone Extraction stopped by user.')
        return jsonify({"message": f"Drone Extraction process successfully stopped"}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": e.stderr.decode("utf-8")}), 500
    
@processing_bp.route('/process_drone_tiff', methods=['POST'])
def process_drone_tiff():
    global processed_image_folder
    global now_drone_processing
    data_root_dir = current_app.config['DATA_ROOT_DIR']
    # receive the parameters
    location = request.json['location']
    population = request.json['population']
    date = request.json['date']
    year = request.json['year']
    experimnent = request.json['experiment']
    platform = request.json['platform']
    sensor = request.json['sensor']

    intermediate_prefix = data_root_dir+'/Intermediate'
    processed_prefix = data_root_dir+'/Processed'
    intermediate_image_folder = os.path.join(intermediate_prefix, year, experimnent, location, population)
    processed_image_folder = os.path.join(processed_prefix, year, experimnent, location, population, date, platform, sensor)
    # print("Setting processed image folder: "+ processed_image_folder)
    # Check if already in processing
    if now_drone_processing:
        return jsonify({'message': 'Already in processing'}), 400
    
    now_drone_processing = True
    shared_states.stop_signal = False
    
    try: 
        rgb_tif_file, dem_tif_file, thermal_tif_file = find_drone_tiffs(processed_image_folder)
        geojson_path = os.path.join(intermediate_image_folder,'Plot-Boundary-WGS84.geojson')
        date = processed_image_folder.split("/")[-3]
        output_geojson = os.path.join(processed_image_folder,f"{date}-{platform}-{sensor}-Traits-WGS84.geojson")
        result = process_tiff(tiff_files_rgb=rgb_tif_file,
                     tiff_files_dem=dem_tif_file,
                     tiff_files_thermal=thermal_tif_file,
                     plot_geojson=geojson_path,
                     output_geojson=output_geojson,
                     debug=False)
        now_drone_processing = False
        shared_states.stop_signal = True
        return jsonify({'message': str(result)}), 200

    except Exception as e:
        now_drone_processing = False
        shared_states.stop_signal = True
        print(e)
        return jsonify({'message': str(e)}), 400


@processing_bp.route('/save_array', methods=['POST'])
def save_array(debug=False):
    data = request.json
    data_root_dir = current_app.config['DATA_ROOT_DIR']
    if 'array' not in data:
        return jsonify({"message": "Missing array in data"}), 400

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
            
            # Check if pointX and pointY are not null
            if item['pointX'] is not None and item['pointY'] is not None:
                # Add or update the point
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
            else:
                # Remove the point if pointX or pointY is null
                if image_name in existing_data:
                    del existing_data[image_name]
                    if debug:
                        print(f"Removed point for image: {image_name}")

    # Write merged data to file
    with open(filename, "w") as f:
        f.write('EPSG:4326\n')
        for image_name, item in existing_data.items():
            formatted_data = f"{item['gcp_lon']} {item['gcp_lat']} 0 {item['pointX']} {item['pointY']} {image_name} {item['gcp_label']} {item['naturalWidth']} {item['naturalHeight']} \n"
            f.write(formatted_data)

    return jsonify({"message": f"Array saved successfully in {filename}!"}), 200

@processing_bp.route('/initialize_file', methods=['POST'])
def initialize_file():
    data = request.json
    data_root_dir = current_app.config['DATA_ROOT_DIR']
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

@processing_bp.route('/query_traits', methods=['POST'])
def query_traits():
    # receive the parameters
    data_root_dir = current_app.config['DATA_ROOT_DIR']
    location = request.json['location']
    population = request.json['population']
    date = request.json['date']
    sensor = request.json['sensor']
    year = request.json['year']
    experiment = request.json['experiment']
    platform = request.json['platform']
    orthomosaic_version = request.json.get('orthomosaic_version')  # Optional parameter for specific version

    prefix = data_root_dir+'/Processed'
    
    # If orthomosaic_version is provided, look in the version-specific directory
    if orthomosaic_version:
        traitpth = os.path.join(prefix, year, experiment, location, population, date, platform, sensor, orthomosaic_version,
                              f"{date}-{sensor}-{orthomosaic_version}-Traits-WGS84.geojson")

    if not os.path.isfile(traitpth):
        return jsonify({'message': []}), 404
    else:
        gdf = gpd.read_file(traitpth)
        traits = list(gdf.columns)
        extraneous_columns = ['Tier','Bed','Plot','Label','Group','geometry']
        traits = [x for x in traits if x not in extraneous_columns]
        print(traits, flush=True)
        return jsonify(traits), 200

@processing_bp.route('/get_orthomosaic_versions', methods=['POST'])
def get_orthomosaic_versions():
    """
    Get available orthomosaic versions for a specific dataset
    Returns both aerial/drone traits (sensor level) and roboflow inference traits (version level)
    """
    data_root_dir = current_app.config['DATA_ROOT_DIR']
    try:
        data = request.json
        year = data.get('year')
        experiment = data.get('experiment')
        location = data.get('location')
        population = data.get('population')
        date = data.get('date')
        platform = data.get('platform')
        sensor = data.get('sensor')
        
        if not all([year, experiment, location, population, date, platform, sensor]):
            return jsonify({"error": "Missing required parameters"}), 400
            
        versions = []
        
        # Check for AgRowStitch versions
        processed_path = os.path.join(data_root_dir, 'Processed', year, experiment, location, population, date, platform, sensor)
        if os.path.exists(processed_path):
            for item in os.listdir(processed_path):
                item_path = os.path.join(processed_path, item)
                if os.path.isdir(item_path) and item.startswith('AgRowStitch_v'):
                    versions.append({'type': 'AgRowStitch', 'AGR_version': item})

        # Check for ODM orthomosaics for splitting if not done in get plot images
        # if os.path.exists(processed_path):
        #     for file in os.listdir(processed_path):
        #         if file.endswith('-RGB.tif'):
        #             versions.append('ODM_Direct')
        #             break
        
        # Check for plot images from split_orthomosaics
        intermediate_path = os.path.join(data_root_dir, 'Intermediate', year, experiment, location, population, 'plot_images', date)
        if os.path.exists(intermediate_path):
            plot_files = [f for f in os.listdir(intermediate_path) if f.startswith('plot_') and f.endswith('.png')]
            if plot_files:
                versions.append({'type': 'Plot_Images', 'AGR_version': 'Plot_Images'})

        prefix = data_root_dir + '/Processed'
        sensor_dir = os.path.join(prefix, year, experiment, location, population, date, platform, sensor)
        
        # versions = []
        
        if os.path.exists(sensor_dir):
            # Check for aerial/drone traits at sensor level
            aerial_trait_file = f"{date}-{platform}-{sensor}-Traits-WGS84.geojson"
            aerial_trait_path = os.path.join(sensor_dir, aerial_trait_file)
            
            if os.path.exists(aerial_trait_path):
                # Convert to relative path for Flask file server
                relative_path = os.path.relpath(aerial_trait_path, data_root_dir)
                versions.append({
                    'type': 'aerial',
                    'version': 'main',
                    'versionName': f'{platform}',
                    'versionType': 'aerial',
                    'path': f'/files/{relative_path.replace(os.sep, "/")}'
                })
            
            # Check for roboflow inference traits in subdirectories
            for item in os.listdir(sensor_dir):
                item_path = os.path.join(sensor_dir, item)
                if os.path.isdir(item_path):
                    # Look for traits file with version-specific naming
                    trait_file = f"{date}-{platform}-{sensor}-{item}-Traits-WGS84.geojson"
                    trait_path = os.path.join(item_path, trait_file)
                    
                    if os.path.exists(trait_path):
                        # Convert to relative path for Flask file server
                        relative_path = os.path.relpath(trait_path, data_root_dir)
                        versions.append({
                            'type': 'roboflow',
                            'version': item,
                            'versionName': f'{item}',
                            'versionType': 'roboflow',
                            'path': f'/files/{relative_path.replace(os.sep, "/")}'
                        })
        print(f"Available orthomosaic versions: {versions}")
        return jsonify(versions), 200
        
    except Exception as e:
        print(f"Error getting orthomosaic versions: {e}")
        return jsonify({'error': str(e)}), 500

@processing_bp.route('/get_agrowstitch_versions', methods=['POST'])
def get_agrowstitch_versions():
    """
    Alias for get_orthomosaic_versions for backward compatibility
    """
    return get_orthomosaic_versions()
    
def select_middle(df):
    middle_index = len(df) // 2  # Find the middle index
    return df.iloc[[middle_index]]  # Use iloc to select the middle row

def filter_images(geojson_features, year, experiment, location, population, date, platform, sensor, middle_image=False):
    data_root_dir = current_app.config['DATA_ROOT_DIR']
    # Construct the CSV path from the state variables
    csv_path = os.path.join(data_root_dir, 'Raw', year, experiment, location, 
                            population, date, 'rover', 'RGB', 'Metadata', 'msgs_synced.csv')
    df = pd.read_csv(csv_path)

    # Convert DataFrame to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))

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
    sensor_col = '/top/' + sensor.lower()
    filtered_images = filtered_gdf[sensor_col+"_file"].tolist()
    filtered_labels = filtered_gdf['Label'].tolist()
    filtered_plots = filtered_gdf['Plot'].tolist()

    # Create a list of dictionaries for the filtered images
    filtered_images_new = []
    for image_name in  filtered_images:
        image_path_abs = os.path.join(data_root_dir, 'Raw', year, experiment, location, population, date, platform, sensor, image_name)
        image_path_rel_to_data_root = os.path.relpath(image_path_abs, data_root_dir)
        filtered_images_new.append(image_path_rel_to_data_root)

    imageDataQuery = [{'imageName': image, 'label': label, 'plot': plot} for image, label, plot in zip(filtered_images_new, filtered_labels, filtered_plots)]

    # Sort the filtered_images by label
    imageDataQuery = sorted(imageDataQuery, key=lambda x: x['label'])

    return imageDataQuery

@processing_bp.route('/query_images', methods=['POST'])
def query_images():
    data_root_dir = current_app.config['DATA_ROOT_DIR']
    data = request.get_json()
    geojson_features = data['geoJSON']
    year = data['selectedYearGCP']
    experiment = data['selectedExperimentGCP']
    location = data['selectedLocationGCP']
    population = data['selectedPopulationGCP']
    date = data['selectedDateQuery']
    sensor = data['selectedSensorQuery']
    middle_image = data['middleImage']
    platform = data['selectedPlatformQuery']

    if platform == 'Drone':
        # Do Drone Image query
        filtered_images = query_drone_images(data,data_root_dir)
    else:
        filtered_images = filter_images(geojson_features, year, experiment, location, 
                                        population, date, platform, sensor, middle_image)

    return jsonify(filtered_images)

@processing_bp.route('/dload_zipped', methods=['POST'])
def dload_zipped():
    data_root_dir = current_app.config['DATA_ROOT_DIR']
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

@processing_bp.route('/save_csv', methods=['POST'])
def save_csv():
    data_root_dir = current_app.config['DATA_ROOT_DIR']
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
    
@processing_bp.route('/save_geojson', methods=['POST'])
def save_geojson():
    data_root_dir = current_app.config['DATA_ROOT_DIR']
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

@processing_bp.route('/load_geojson', methods=['POST'])
def load_geojson():
    data_root_dir = current_app.config['DATA_ROOT_DIR']
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
    
@processing_bp.route('/get_odm_logs', methods=['POST'])
def get_odm_logs():
    data_root_dir = current_app.config['DATA_ROOT_DIR']
    # get data
    data = request.json
    method = data.get('method')
    ortho_data = data.get('orthoData')
    
    print("Method for logs:", method)  # Debug statement
    
    if method == 'STITCH':
        final_mosaics_path = os.path.join(
            data_root_dir, "Raw", ortho_data['year'],
            ortho_data['experiment'], ortho_data['location'], ortho_data['population'],
            ortho_data['date'], ortho_data['platform'], ortho_data['sensor'], 
            "Images", "final_mosaics"
        )
        
        if not os.path.exists(final_mosaics_path):
            return jsonify({"error": "Final mosaics directory not found"}), 404
        
        # Find all plot log files
        log_files = []
        for file in os.listdir(final_mosaics_path):
            if file.startswith("temp_plot_") and file.endswith(".log"):
                log_files.append(os.path.join(final_mosaics_path, file))
        
        if not log_files:
            return jsonify({"error": "No plot log files found"}), 404
        
        # Sort log files by plot ID for consistent order
        log_files.sort()
        
        # Combine logs from all plots
        combined_logs = []
        for log_file in log_files:
            plot_name = os.path.basename(log_file).replace(".log", "")
            combined_logs.append(f"\n=== {plot_name.upper()} LOG ===\n")
            
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    combined_logs.append(content)
                    if not content.endswith('\n'):
                        combined_logs.append('\n')
            except Exception as e:
                combined_logs.append(f"Error reading {log_file}: {str(e)}\n")
        
        return jsonify({"log_content": ''.join(combined_logs)}), 200
    else:
        # Original ODM logs logic
        logs_path = os.path.join(data_root_dir, 'temp', 'project', 'code', 'logs.txt')
        print("Logs Path:", logs_path)  # Debug statement
        if os.path.exists(logs_path):
            with open(logs_path, 'r') as log_file:
                lines = log_file.readlines()
                # latest_logs = lines[-20:]  # Get the last 20 lines
            return jsonify({"log_content": ''.join(lines)}), 200
        else:
            print("Logs not found at path:", logs_path)  # Debug statement
            return jsonify({"error": "Logs not found"}), 404

@processing_bp.route('/run_stitch', methods=['POST'])
def run_stitch_endpoint():
    data_root_dir = current_app.config['DATA_ROOT_DIR']
    data = request.json
    location = data.get('location')
    population = data.get('population')
    date = data.get('date')
    year = data.get('year')
    experiment = data.get('experiment')
    platform = data.get('platform')
    sensor = data.get('sensor')
    # temp_dir = data.get('temp_dir')
    # reconstruction_quality = data.get('reconstruction_quality')
    custom_options = data.get('custom_options')
    
    global stitch_thread, stitch_stop_event, odm_method
    global stitched_path, temp_output
    stitch_thread = current_app.config['STITCH_THREAD']
    stitch_stop_event = current_app.config['STITCH_STOP_EVENT']

    odm_method = 'stitch'
    current_app.config['ODM_METHOD'] = odm_method
    stitch_stop_event.clear()
    
    try:
        image_path = os.path.join(
            data_root_dir, "Raw", year,
            experiment, location, population,
            date, platform, sensor, "Images", "top"
        )
        msgs_synced_path = os.path.join(
            data_root_dir, "Raw", year,
            experiment, location, population,
            date, platform, sensor, "Metadata", "msgs_synced.csv"
        )
        save_path = os.path.join(
            data_root_dir, "Processed", year,
            experiment, location, population,
            date, platform, sensor
        )
        image_calibration = os.path.join(
            data_root_dir, "Raw", year,
            experiment, location, population,
            date, platform, sensor, "Metadata", "top_calibration.json"
        )
        config_path = f"{AGROWSTITCH_PATH}/panorama_maker/config.yaml"
        stitched_path = os.path.join(os.path.dirname(image_path), "final_mosaics")
        temp_output = os.path.join(os.path.dirname(image_path), "top_output")
        
        # Check if image_path exists
        if not os.path.exists(image_path):
            return jsonify({
                "status": "error", 
                "message": f"Error: Image path not found at {image_path}\n\nThe selected images may not be compatible with AgRowStitch yet."
            }), 404
        
        # remove stitched_path and temp_output if it exists
        if os.path.exists(stitched_path):
            shutil.rmtree(stitched_path)
            print(f"Removed existing stitched path: {stitched_path}")
        if os.path.exists(temp_output):
            shutil.rmtree(temp_output)
            print(f"Removed existing temp output path: {temp_output}")
        
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        
        # Check if msgs_synced.csv exists
        if not os.path.exists(msgs_synced_path):
            return jsonify({"status": "error", "message": f"msgs_synced.csv not found at {msgs_synced_path}"}), 404
        
        # Load and validate plot indices
        import pandas as pd
        msgs_df = pd.read_csv(msgs_synced_path)
        
        # Check if plot_index column exists
        if 'plot_index' not in msgs_df.columns:
            return jsonify({
                "status": "error", 
                "message": "Plot index column not found in msgs_synced.csv. Please perform plot marking in File Management first."
            }), 400
        
        # Check if any plot indices are defined (excluding plot 0 which is usually background/unassigned)
        unique_plots = [pid for pid in msgs_df['plot_index'].unique() if pid != 0 and not pd.isna(pid)]
        
        if len(unique_plots) == 0:
            return jsonify({
                "status": "error", 
                "message": "No plot indices are defined in the dataset. Please perform plot marking in File Management to assign images to plots before running AgRowStitch."
            }), 400
        
        print(f"Found {len(unique_plots)} unique plots for stitching: {unique_plots}")
        
        # Create a monitoring stop event to coordinate threads
        monitoring_stop_event = threading.Event()
        
        # Start multi-plot log monitoring thread
        final_mosaics_dir = os.path.join(os.path.dirname(image_path), "final_mosaics")
        
        def progress_callback(progress):
            current_app.config['LATEST_DATA']['ortho'] = progress

        thread_prog = threading.Thread(
            target=monitor_stitch_updates_multi_plot, 
            args=(final_mosaics_dir, unique_plots, progress_callback, monitoring_stop_event), 
            daemon=True
        )
        thread_prog.start()
        
        # Start the stitching process for all plots in background
        stitch_thread = threading.Thread(
            target=complete_stitch_workflow,
            args=(msgs_synced_path, image_path, config_path, custom_options, 
                  save_path, image_calibration, stitch_stop_event, progress_callback, monitoring_stop_event),
            daemon=True
        )
        stitch_thread.start()

        return jsonify({"status": "started", "message": "Stitching process started for all plots"}), 202

    except Exception as e:
        print(f"Error running AgRowStitch: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@processing_bp.route('/run_odm', methods=['POST'])
def run_odm_endpoint():
    latest_data = current_app.config['LATEST_DATA']
    data_root_dir = current_app.config['DATA_ROOT_DIR']
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
    args = make_odm_args(data_root_dir, 
                         location, 
                         population, 
                         date, 
                         year, 
                         experiment, 
                         platform, 
                         sensor, 
                         temp_dir, 
                         reconstruction_quality, 
                         custom_options)
    try:
        # Check if the container exists
        command = f"docker ps -a --filter name=GEMINI-Container --format '{{{{.Names}}}}'"
        command = command.split()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        container_exists = "GEMINI-Container" in stdout.decode().strip()

        if container_exists:
            print('Removing temp folder...')
            folder_to_delete = os.path.join(data_root_dir, 'temp', 'project')
            cleanup_command = f"docker exec GEMINI-Container rm -rf {folder_to_delete}"
            cleanup_command = cleanup_command.split()
            cleanup_process = subprocess.Popen(cleanup_command, stderr=subprocess.STDOUT)
            cleanup_process.wait()
            
            # Stop the container if it's running
            command = f"docker stop GEMINI-Container"
            command = command.split()
            process = subprocess.Popen(command, stderr=subprocess.STDOUT)
            process.wait()

            # Remove the container if it exists
            command = f"docker rm GEMINI-Container"
            command = command.split()
            process = subprocess.Popen(command, stderr=subprocess.STDOUT)
            process.wait()

        # # Proceed with the reset and starting threads
        # reset_thread = threading.Thread(target=reset_odm, args=(args,), daemon=True)
        # reset_thread.start()
        # reset_thread.join()  # Ensure reset thread is finished before proceeding
        
        # Run ODM in a separate thread
        thread = threading.Thread(target=run_odm, args=(args,), daemon=True)
        thread.start()
        
        # Run progress tracker
        logs_path = os.path.join(data_root_dir, 'temp/project/code/logs.txt')
        progress_file = os.path.join(data_root_dir, 'temp/progress.txt')
        os.makedirs(os.path.dirname(progress_file), exist_ok=True)
        thread_prog = threading.Thread(target=monitor_log_updates, args=(latest_data, logs_path, progress_file), daemon=True)
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
        return make_response(jsonify({"status": "error", "message": f"ODM processing failed to start {str(e)}"}), 404)


@processing_bp.route('/stop_odm', methods=['POST'])
def stop_odm():
    data_root_dir = current_app.config['DATA_ROOT_DIR']
    # global data_root_dir, stitch_thread, stitch_stop_event, odm_method, stitched_path, temp_output
    try:
        if odm_method == 'stitch' and stitch_thread is not None:
            print("Stopping stitching thread...")
            stitch_stop_event.set()
            stitch_thread.join(timeout=10)  # Wait up to 10s for clean exit
            print("Stitching stopped.")
            
            # remove stitched_path and temp_output if it exists
            if os.path.exists(stitched_path):
                shutil.rmtree(stitched_path)
                print(f"Removed existing stitched path: {stitched_path}")
            if os.path.exists(temp_output):
                shutil.rmtree(temp_output)
                print(f"Removed existing temp output path: {temp_output}")
        else:
            print('ODM processed stopped by user.')
            print('Removing temp folder...')
            folder_to_delete = os.path.join(data_root_dir, 'temp', 'project')
            cleanup_command = f"docker exec GEMINI-Container rm -rf {folder_to_delete}"
            cleanup_command = cleanup_command.split()
            cleanup_process = subprocess.Popen(cleanup_command, stderr=subprocess.STDOUT)
            cleanup_process.wait()
            
            print('Stopping ODM process...')
            stop_event = threading.Event()
            stop_event.set()
            command = f"docker stop GEMINI-Container"
            command = command.split()
            # Run the command
            process = subprocess.Popen(command, stderr=subprocess.STDOUT)
            process.wait()
            
            print('Removing ODM container...')
            command = f"docker rm GEMINI-Container"
            command = command.split()
            # Run the command
            process = subprocess.Popen(command, stderr=subprocess.STDOUT)
            process.wait()
            # reset_odm(data_root_dir)
            shutil.rmtree(os.path.join(data_root_dir, 'temp'))
        return jsonify({"message": "ODM process stopped"}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": e.stderr.decode("utf-8")}), 500