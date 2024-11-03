# Standard library imports
import os
import shutil
import subprocess
import sys
import argparse
from glob import glob
from math import log
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime

# Third-party library imports
import cv2
from numpy import std
from tqdm import tqdm
import yaml
import json

# Local application/library specific imports
# Add script directory to path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from create_geotiff_pyramid import create_tiled_pyramid
from thermal_camera.flir_one_pro_extract import extract_thermal_images_NIRFormat as extract_thermal_images
from utils import check_nvidia_smi

def _create_directory_structure(args):
    
    '''
    Create a temporary directory structure for ODM and arrange the images and
    gcp_list.txt in the appropriate locations.
    '''

    # Create the temporary directory structure
    pth = args.temp_dir

    if pth[-1] == '/':
        pth = pth[:-1]
    if os.path.basename(pth) != 'project':
        pth = os.path.join(pth, 'project')

    if not os.path.exists(pth):
        os.makedirs(pth)
        os.makedirs(os.path.join(pth, 'code'))


    # Copy the gcp_list.txt to the temporary directory
    gcp_pth = os.path.join(args.data_root_dir, 'Intermediate', args.year, args.experiment, args.location, args.population, args.date, args.platform, args.sensor, 'gcp_list.txt')
    # print(f"GCP Path: {gcp_pth}")
    # Check if gcp_pth exists
    if not os.path.exists(gcp_pth):
        print(f"GCP file {gcp_pth} does not exist.")
        return False
    else:
        # Check if the file exists but has more than 1 lines
        filtered_gcp_list = []
        with open(gcp_pth, 'r') as f:
            lines = f.readlines()
            if len(lines) < 2:
                print("GCP file has less than 2 lines...Ignoring")
            else:
                shutil.copy(gcp_pth, os.path.join(pth, 'code', 'gcp_list.txt'))

def _process_outputs(args):

    pth = args.temp_dir

    if pth[-1] == '/':
        pth = pth[:-1]
    if os.path.basename(pth) != 'project':
        pth = os.path.join(pth, 'project')    

    ortho_file = os.path.join(pth, 'code', 'odm_orthophoto', 'odm_orthophoto.tif')
    dem_file = os.path.join(pth, 'code', 'odm_dem', 'dsm.tif')

    output_folder = os.path.join(args.data_root_dir, 'Processed', args.year, args. experiment, args.location, args.population, args.date, args.platform, args.sensor)
    print(f"Output folder: {output_folder}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Copy the orthomosaic and DEM to the output folder
    # Check if the files exist
    if os.path.exists(ortho_file):
        output_path = os.path.join(output_folder, args.date+'-RGB.tif')
        os.system(f'cp {ortho_file} {output_path}')
    else:
        print(f"Error: Orthomosaic file {ortho_file} does not exist.")
        return False

    if os.path.exists(dem_file):
        output_path = os.path.join(output_folder, args.date+'-DEM.tif')
        os.system(f'cp {dem_file} {output_path}')
    else:
        print(f"Error: DEM file {dem_file} does not exist.")
        return False
    # Process pyramids
    print("Processing pyramids")
    create_tiled_pyramid(ortho_file, os.path.join(output_folder, args.date+'-RGB-Pyramid.tif'))
    create_tiled_pyramid(dem_file, os.path.join(output_folder, args.date+'-DEM-Pyramid.tif'))

    # Copy the metadata file to the output folder
    metadata_file = args.metadata_file.split('/')[-1]
    shutil.copy(args.metadata_file, os.path.join(output_folder, metadata_file))

    # Move ODM outputs to /var/tmp so they can be accessed if needed but deleted eventually
    # if not os.path.exists(os.path.join('/var/tmp', args.location, args.population, args.date, args.sensor)):
    #     os.makedirs(os.path.join('/var/tmp', args.location, args.population, args.date, args.sensor))
    # shutil.move(os.path.join(pth, 'code'), os.path.join('/var/tmp', args.location, args.population, args.date, args.sensor))
    
    # Delete the temporary directory
    # shutil.rmtree(pth)
    print("Processing complete.")
    return True

def make_odm_args_from_metadata(metadata):
    '''
    Generate an argparse.Namespace object from the metadata file.
    Example of yaml_data:
    
    '''
    year = metadata['year']
    experiment = metadata['experiment']
    location = metadata['location']
    population = metadata['population']
    date = metadata['date']
    platform = metadata['platform']
    sensor = metadata['sensor']
    reconstruction_quality = metadata['reconstruction_quality']
    custom_options = metadata['custom_options']
    image_pth = metadata['image_pth']
    data_root_dir = metadata['data_root_dir']
    temp_dir = metadata['temp_dir']

    return make_odm_args(data_root_dir, location, population, date, year, experiment, platform, sensor, temp_dir, reconstruction_quality, custom_options)

def make_odm_args(data_root_dir, location, population, date, year, experiment, platform, sensor, temp_dir, reconstruction_quality, custom_options):
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

    return args

def reset_odm(args, metadata_file_name=None):
    # May be more appropriate to rename as "prep_odm" in the future to clarify functionality
    # This function now contains checks for existing processed data previously in run_odm
    drd = args.data_root_dir
    temp_dir = args.temp_dir

    # Check if the temp directory exists
    if not os.path.exists(temp_dir):
        # No need to reset ODM
        return

    if metadata_file_name is None:
        metadata_file_name = 'metadata.json'
        
    metadata_file = os.path.join(temp_dir, 'code', metadata_file_name)
    # Check if the metadata file exists
    if not os.path.exists(metadata_file):
        # Try with yaml
        metadata_file = os.path.join(temp_dir, 'code', metadata_file_name.replace('.json', '.yaml'))
        
    reset_odm_temp = False
    if os.path.exists(metadata_file):
        # Read the metadata file
        with open(metadata_file, 'r') as file:
            if metadata_file.endswith('.json'):
                data = json.load(file)
            elif metadata_file.endswith('.yaml'):
                data = yaml.load(file, Loader=yaml.FullLoader)
            else:
                raise ValueError('Invalid metadata file format. Must be either .yaml or .json')
            
            image_pth = data['image_pth']
            prev_arg = make_odm_args_from_metadata(data)
            if odm_args_checker(prev_arg, args):
                print("Already processed. Try Copying to Processed folder.")
                if _process_outputs(args) == True:
                    return
                else:
                    reset_odm_temp = True
            else:
                # Reset the ODM if the arguments are different
                reset_odm_temp = True
    else:
        reset_odm_temp = True
    
    if reset_odm_temp:
        shutil.rmtree(temp_dir)

def odm_args_checker(arg1, arg2):
    """
    Check if two argparse.Namespace objects are equal based on certain attributes.
    """
    return all([
        arg1.location == arg2.location,
        arg1.population == arg2.population,
        arg1.date == arg2.date,
        arg1.platform == arg2.platform,
        arg1.sensor == arg2.sensor
    ])

def run_odm(args):
    '''
    Run ODM on the temporary directory.
    '''
    metadata_file_name = 'ortho_metadata.json'
    project_path = args.temp_dir
    metadata_file = os.path.join(project_path, 'code', metadata_file_name)
    args.metadata_file = metadata_file
    try:
        # Reset ODM if the arguments are different
        reset_odm(args, metadata_file_name=metadata_file_name)
    except Exception as e:
        # Handle exception: log it, set a flag, etc.
        print(f"Error in thread: {e}")
        return
    
    try:
        _create_directory_structure(args)

        image_pth = os.path.join(args.data_root_dir, 'Raw', args.year, args.experiment, args.location, args.population, args.date, args.platform, args.sensor, 'Images')        # Check if the sensor is thermal
        if args.sensor.lower() == 'thermal':
            # Extract thermal images if the sensor is thermal
            extracted_folder_name = os.path.join('Images_extracted')
            extract_thermal_images(image_pth, extracted_folder_name)
            image_pth = image_pth.replace('Images', extracted_folder_name)

            # Copy geo.txt file to the temporary directory
            geo_txt_path = image_pth.replace('Images_extracted', 'geo.txt')

            if os.path.exists(geo_txt_path):
                shutil.copy(geo_txt_path, os.path.join(args.temp_dir, 'code', 'geo.txt'))

        # Run ODM
        pth = args.temp_dir

        if pth[-1] == '/':
            pth = pth[:-1]
        if os.path.basename(pth) != 'project':
            pth = os.path.join(pth, 'project')
        print('Project Path: ', pth)
        image_pth = os.path.join(args.data_root_dir, 'Raw', args.year, args.experiment, args.location, args.population, args.date, args.platform, args.sensor, 'Images')
        print('Image Path: ', image_pth)
        options = ""
        log_file = os.path.join(project_path, 'code', 'logs.txt')
        with open(log_file, 'w') as f:
            # See options from https://docs.opendronemap.org/arguments/
            common_options = "--dsm" # orthophoto-resolution gsd is usually 0.27cm/pixel
            reconstruction_quality = args.reconstruction_quality.lower()
            if reconstruction_quality == 'custom':
                options = f"{args.custom_options} {common_options}"
                print('Starting ODM with custom options...')
            elif reconstruction_quality == 'low':
                options = f"{common_options} --pc-quality medium --min-num-features 8000 --orthophoto-resolution 2.0" 
                print('Starting ODM with low options...')
            elif reconstruction_quality == 'lowest':
                options = f"{common_options} --fast-orthophoto "
                print('Starting ODM with Lowest options...')
            elif reconstruction_quality == 'high':
                options = f"{common_options} --pc-quality high --min-num-features 16000 --orthophoto-resolution 0.5"
                print('Starting ODM with high options...')
            else:
                raise ValueError('Invalid reconstruction quality: {}. Must be one of: low, high, custom'.format(args.reconstruction_quality))
            
            if args.sensor.lower() == 'thermal':
                #options += ' --radiometric-calibration camera'
                pass
            # Create the command
            # It will mount pth to /datasets and image_pth to /datasets/code/images
            volumes = f"-v {project_path}:/datasets -v {image_pth}:/datasets/code/images -v /etc/timezone:/etc/timezone:ro -v /etc/localtime:/etc/localtime:ro"
            if check_nvidia_smi():
                docker_image = "--gpus all opendronemap/odm:gpu"
            else:
                docker_image = "opendronemap/odm"
            # Create a container name with the project name
            container_name = f"GEMINI-Container-{args.location}-{args.population}-{args.date}-{args.sensor}"
            command = f"docker run --name {container_name} -i --rm {volumes} {docker_image} --project-path /datasets code {options}" # 'code' is the default project name
            # command = f"docker run --user {user_id}:{group_id} --name GEMINI-Container -i --rm {volumes} {docker_image} --project-path /datasets code {options}"
            # user_id = os.getenv("UID", os.getuid())  # Get the current user ID
            # group_id = os.getenv("GID", os.getgid())  # Get the current group ID
            # Save image_pth and  docker command to metadata yaml file
            metadata_file = os.path.join(project_path, 'code', metadata_file_name)
            with open(metadata_file, 'w') as file:
                # Export arg to data
                metadata = vars(args)
                metadata['image_pth'] = image_pth
                metadata['command'] = command
                metadata['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M")
                # yaml.dump(data, file, sort_keys=False)
                json.dump(metadata, file)
                args.metadata_file = metadata_file

            # Parse this with space to list
            command = command.split()
            # Run the command
            process = subprocess.Popen(command,stdout=f, stderr=subprocess.STDOUT)
            process.wait()
        
        _process_outputs(args)
        
    
    except Exception as e:
        # Handle exception: log it, set a flag, etc.
        print(f"Error in thread: {e}")
    
if __name__ == '__main__':
    # Main function for debugging
    parser = argparse.ArgumentParser(description='Generate an orthomosaic for a set of images')
    parser.add_argument('--year', type=str, help='Year of the data collection',default='2023')
    parser.add_argument('--experiment', type=str, help='Experiment name', default='Davis')
    parser.add_argument('--location', type=str, help='Location of the data collection', default='Davis')
    parser.add_argument('--population', type=str, help='Population for the dataset', default='Legumes')
    parser.add_argument('--date', type=str, help='Date of the data collection', default='2023-07-18')
    parser.add_argument('--platform', type=str, help='Platform used', default='Drone')
    parser.add_argument('--sensor', type=str, help='Sensor used', default='thermal')
    parser.add_argument('--data_root_dir', type=str, help='Root directory for the data', default='/home/GEMINI/GEMINI-App-Data')
    parser.add_argument('--temp_dir', type=str, help='Temporary directory to store the images and gcp_list.txt',
                        default='/home/GEMINI/temp/project')
    parser.add_argument('--data_root_dir', type=str, help='Root directory for the data', default='/home/GEMINI/GEMINI-Data')
    parser.add_argument('--reconstruction_quality', type=str, help='Reconstruction quality (high, low, custom)',
                        choices=[' high', 'low', 'lowest', 'custom'], default='lowest')
    parser.add_argument('--custom_options', nargs='+', help='Custom options for ODM (e.g. --orthophoto-resolution 0.01)', 
                        required=False)
    args = parser.parse_args()

    # Check if opendronemap is installed as a snap package
    if shutil.which('docker') is None:
        print("Error: docker is not installed. Please install docker")
        sys.exit(1)
    
    # Run ODM
    run_odm(args)