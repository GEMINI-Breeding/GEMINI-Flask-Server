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
import rasterio
import numpy as np
import io
from datetime import datetime
from PIL import Image
from PIL.Image import Resampling

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

def _copy_image(src_folder, dest_folder, image_name):
    
    src_path = os.path.join(src_folder, image_name)
    dest_path = os.path.join(dest_folder, image_name)

    if not os.path.exists(dest_path):
        try:
            shutil.copy(src_path, dest_path)
        except PermissionError:
            # Fallback to using system cp command if shutil.copy fails
            os.system(f'cp "{src_path}" "{dest_path}"')

def _create_directory_structure(args):
    
    '''
    Create a temporary directory structure for ODM and arrange the images and
    gcp_list.txt in the appropriate locations.
    '''

    # Create the temporary directory structure
    project_path = args.temp_dir

    if project_path[-1] == '/':
        project_path = project_path[:-1]
    if os.path.basename(project_path) != 'project':
        project_path = os.path.join(project_path, 'project')

    if not os.path.exists(project_path):
        os.makedirs(project_path)
        os.makedirs(os.path.join(project_path, 'code'))


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
                try:
                    shutil.copy(gcp_pth, os.path.join(project_path, 'code', 'gcp_list.txt'))
                except PermissionError:
                    os.system(f'cp "{gcp_pth}" "{os.path.join(project_path, "code", "gcp_list.txt")}"')

def _process_outputs(args):

    project_path = args.temp_dir

    if project_path[-1] == '/':
        project_path = project_path[:-1]
    if os.path.basename(project_path) != 'project':
        project_path = os.path.join(project_path, 'project')    

    ortho_file = os.path.join(project_path, 'code', 'odm_orthophoto', 'odm_orthophoto.tif')
    dem_file = os.path.join(project_path, 'code', 'odm_dem', 'dsm.tif')

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
    try:
        shutil.copy(args.metadata_file, os.path.join(output_folder, metadata_file))
    except PermissionError:
        os.system(f'cp "{args.metadata_file}" "{os.path.join(output_folder, metadata_file)}"')

    # Move ODM outputs to /var/tmp so they can be accessed if needed but deleted eventually
    temp_folder = os.path.join('/var/tmp', args.location, args.population, args.date, args.sensor)
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    shutil.move(os.path.join(project_path, 'code'), temp_folder)
    
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
                _process_outputs(args)
                print("Already processed. Copying to Processed folder.")
                return
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

def convert_tif_to_png(tif_path):
    
    Image.MAX_IMAGE_PIXELS = None
    
    try:
        directory = os.path.dirname(tif_path)
        filename = os.path.basename(tif_path)
        filename_without_ext = os.path.splitext(filename)[0]
        png_path = os.path.join(directory, filename_without_ext + '.png')
        
        with Image.open(tif_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            max_size = (1024, 1024)  # Example maximum dimensions
            img.thumbnail(max_size, Resampling.LANCZOS)
            
            img.save(png_path, 'PNG')
            print(f"Successfully converted {tif_path} to {png_path}")
            
    except Exception as e:
        print(f"Error saving tif as png: {str(e)}")
        print(f"Error saving tif as png: {e}")

def run_odm(args):
    '''
    Run ODM on the temporary directory.
    See options from https://docs.opendronemap.org/arguments/

    Possible considerations & Options:
    ```bash
    --fast-orthophoto
    --dsm --orthophoto-resolution 2.0 --sfm-algorithm planar
    --dsm --orthophoto-resolution 0.01
    --feature-type sift # But its slower?
    --dsm --dem-resolution 0.03 --orthophoto-resolution 0.03
    ```
    
    Notes:
    - GEMINI DJI P4 10m GSD is usually 0.27cm/pixel
    '''
    
    try:
        _create_directory_structure(args)

        # Run ODM
        project_path = args.temp_dir

        if project_path[-1] == '/':
            project_path = project_path[:-1]
        if os.path.basename(project_path) != 'project':
            project_path = os.path.join(project_path, 'project')
        
        
        print('Project Path: ', project_path)
        image_pth = os.path.join(args.data_root_dir, 'Raw', args.year, args.experiment, args.location, args.population, args.date, args.platform, args.sensor, 'Images')
        print('Image Path: ', image_pth)
        odm_options = ""
        log_file = os.path.join(project_path, 'code', 'logs.txt')
        with open(log_file, 'w') as f:
            base_options = ""
            
            # TODO: Add if statement for larger dataset. len(images) > 1000
            base_options += " --feature-quality low --matcher-neighbors 10 --pc-quality low --dsm"

            if args.reconstruction_quality == 'Custom':
                odm_options = f"{base_options} {args.custom_options} "
                print('Starting ODM with custom options...')
            elif args.reconstruction_quality == 'Default':
                odm_options = f"{base_options}" 
                print('Starting ODM with default options...')
            else:
                raise ValueError('Invalid reconstruction quality: {}. Must be one of: default, custom'.format(args.reconstruction_quality))
            
            if args.sensor.lower() == 'thermal':
                #options += ' --radiometric-calibration camera'
                pass

            # It will mount pth to /datasets and image_pth to /datasets/code/images
            volumes = f"-v {project_path}:/datasets -v {image_pth}:/datasets/code/images -v /etc/timezone:/etc/timezone:ro -v /etc/localtime:/etc/localtime:ro"
            
            # Create a container name with the project name
            container_name = f"GEMINI-Container-{args.location}-{args.population}-{args.date}-{args.sensor}"

            if check_nvidia_smi():
                docker_image = "--gpus all opendronemap/odm:gpu"
            else:
                docker_image = "opendronemap/odm"

            # Create the command
            command = f"docker run --name {container_name} -i --rm {volumes} {docker_image} \
                        --gcp /datasets/code/gcp_list.txt  \
                        --project-path /datasets code {odm_options}" # 'code' is the default project name
            
            # Save image_pth and  docker command to recipe yaml file
            data = {
                'year': args.year,
                'experiment': args.experiment,
                'location': args.location,
                'population': args.population,
                'date': args.date,
                'platform': args.platform,
                'sensor': args.sensor,
                'reconstruction_quality': args.reconstruction_quality,
                'custom_options': args.custom_options,
                'image_pth': image_pth,
                'command': command,
            }
            recipe_file = os.path.join(project_path, 'code', 'recipe.yaml')
            with open(recipe_file, 'w') as file:
                yaml.dump(data, file, sort_keys=False)

            # Parse this with space to list
            command = command.split()
            # Run the command
            process = subprocess.Popen(command,stdout=f, stderr=subprocess.STDOUT)
            process.wait()
        
        _process_outputs(args)
        
        # save png image
        convert_tif_to_png(os.path.join(args.data_root_dir, 'Processed', args.year, args.experiment, args.location, args.population, args.date, args.platform, args.sensor, args.date+'-RGB.tif'))
    
    except Exception as e:
        # Handle exception: log it, set a flag, etc.
        print(f"Error in thread: {e}")
    
if __name__ == '__main__':
    # Main function for debugging
    parser = argparse.ArgumentParser(description='Generate an orthomosaic for a set of images')
    parser.add_argument('--date', type=str, help='Date of the data collection')
    parser.add_argument('--location', type=str, help='Location of the data collection')
    parser.add_argument('--population', type=str, help='Population for the dataset')
    parser.add_argument('--year', type=str, help='Year of the data collection')
    parser.add_argument('--experiment', type=str, help='Experiment name')
    parser.add_argument('--sensor', type=str, help='Sensor used')
    parser.add_argument('--temp_dir', type=str, help='Temporary directory to store the images and gcp_list.txt',
                        default='/home/GEMINI/GEMINI-App-Data/temp/project') # TODO: Automatically generate a temp directory? or use /var/tmp?
    parser.add_argument('--reconstruction_quality', type=str, help='Reconstruction quality (default, custom)',
                        choices=['Default', 'Custom'], default='default')
    parser.add_argument('--custom_options', nargs='+', help='Custom options for ODM (e.g. --orthophoto-resolution 0.01)', 
                        required=False)
    args = parser.parse_args()

    # Check if opendronemap is installed as a snap package
    if shutil.which('docker') is None:
        print("Error: docker is not installed. Please install docker")
        sys.exit(1)
    
    # Run ODM
    run_odm(args)