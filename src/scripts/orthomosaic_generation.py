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

# Third-party library imports
import cv2
from numpy import std
from tqdm import tqdm
import yaml

# Local application/library specific imports
# Add script directory to path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from create_geotiff_pyramid import create_tiled_pyramid
from thermal_camera.flir_image_extractor import FlirImageExtractor
from thermal_camera.process_flir import warp_image

def _copy_image(src_folder, dest_folder, image_name):
    
    src_path = os.path.join(src_folder, image_name)
    dest_path = os.path.join(dest_folder, image_name)

    if not os.path.exists(dest_path):
        shutil.copy(src_path, dest_path)

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


    # Copy the recipe file to the output folder
    shutil.copy(os.path.join(pth, 'code', 'recipe.yaml'), os.path.join(output_folder, 'recipe.yaml'))

    # Move ODM outputs to /var/tmp so they can be accessed if needed but deleted eventually
    # if not os.path.exists(os.path.join('/var/tmp', args.location, args.population, args.date, args.sensor)):
    #     os.makedirs(os.path.join('/var/tmp', args.location, args.population, args.date, args.sensor))
    # shutil.move(os.path.join(pth, 'code'), os.path.join('/var/tmp', args.location, args.population, args.date, args.sensor))
    
    # Delete the temporary directory
    # shutil.rmtree(pth)
    print("Processing complete.")
    return True




def check_nvidia_smi():
    '''
    Check if nvidia-smi is installed on the system.
    '''
    # Check the output of "docker run --rm --gpus all nvidia/cuda:11.0.3-base nvidia-smi"
    try:
        output = subprocess.check_output(['docker', 'run', '--rm', '--gpus', 'all', 'nvidia/cuda:11.0.3-base', 'nvidia-smi'])
        if 'NVIDIA-SMI' in output.decode('utf-8'):
            return True
        else:
            return False
    except Exception as e:
        return False

def make_odm_args_from_recipe(yaml_data):
    '''
    Generate an argparse.Namespace object from the recipe.yaml file.
    Example of yaml_data:
    
    '''
    year = yaml_data['year']
    experiment = yaml_data['experiment']
    location = yaml_data['location']
    population = yaml_data['population']
    date = yaml_data['date']
    platform = yaml_data['platform']
    sensor = yaml_data['sensor']
    reconstruction_quality = yaml_data['reconstruction_quality']
    custom_options = yaml_data['custom_options']
    image_pth = yaml_data['image_pth']
    data_root_dir = yaml_data['data_root_dir']
    temp_dir = yaml_data['temp_dir']

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

def reset_odm(args):
    # May be more appropriate to rename as "prep_odm" in the future to clarify functionality
    # This function now contains checks for existing processed data previously in run_odm
    drd = args.data_root_dir
    pth = args.temp_dir
    recipe_file = os.path.join(pth, 'code', 'recipe.yaml')
    reset_odm_temp = False
    if os.path.exists(recipe_file):
        # Read the recipe file
        with open(recipe_file, 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            image_pth = data['image_pth']
            prev_arg = make_odm_args_from_recipe(data)
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
        temp_path = os.path.join(drd, 'temp')
        while os.path.exists(temp_path):
            shutil.rmtree(temp_path)

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

def extract_thermal_images(image_pth, extracted_folder_name):
    '''
    # Extract thermal images
    '''

    fie = FlirImageExtractor(exiftool_path="exiftool",use_calibration_data=True, is_debug=False)

    # Create the folder
    extracted_folder = os.path.join(image_pth, "../",extracted_folder_name)
    if not os.path.exists(extracted_folder):
        os.makedirs(extracted_folder)

    
    # Get the list of images
    images = glob(os.path.join(image_pth, '*.jpg'))

    # Prepare geo.txt file
    geo_txt_path = os.path.join(image_pth, "../", 'geo.txt')
    with open(geo_txt_path, "w") as f:
        # Write the projection to the file
        f.write("EPSG:4326\n")

        # Extract the thermal images
        print("Extracting thermal images...")
        for i, image in enumerate(tqdm(images)):
            image_name = os.path.basename(image)
            # Change the extension to .tiff
            image_name = image_name.replace('.jpg', '.tif')

            # Check if the image is already extracted
            if os.path.exists(os.path.join(extracted_folder, image_name)):
                continue

            fie.process_image(image)
            # Extract Thermal Image
            raw_rgb = fie.get_rgb_np()
            thermal_image_np = fie.get_RawThermalImage()
            thermal_image_np = thermal_image_np.astype('uint16')
            # Warp RGB
            warped_rgb, _  = warp_image(raw_rgb, thermal_image_np, fie.meta, obj_distance=10.0)

            # Convert into uint16
            warped_rgb_uint16 = (warped_rgb * 2**8).astype('uint16')

            # Make RGB-Thermal 4 channel image
            rgb_thermal = cv2.merge([warped_rgb_uint16[:,:,0], warped_rgb_uint16[:,:,1], warped_rgb_uint16[:,:,2], thermal_image_np])

            # Save the image

            cv2.imwrite(os.path.join(extracted_folder, image_name), rgb_thermal)

            # Dry run
            # if i > 10:
            #     break

            # Write the geo.txt file
            gps_info = fie.extract_gps_info()
            gps_info_keys = gps_info.keys()
            # Check if the GPS info is available
            if 'GPSLatitude' in gps_info_keys and 'GPSLongitude' in gps_info_keys and 'GPSAltitude' in gps_info_keys:
                lat = gps_info['latitude']
                lon = gps_info['longitude']
                height = gps_info['altitude']
                f.write(f"{image_name} {lon} {lat} {height}\n")

    print("Thermal images extracted.")





def run_odm(args):
    '''
    Run ODM on the temporary directory.
    '''
    
    # Check if the already processed data is not in the Processed folder
    # Copy the temp to the Processed folder without deleting the temp, and finish
    # Check if the log file exists
    project_path = args.temp_dir
    recipe_file = os.path.join(project_path,'recipe.yaml')
    reset_odm_temp = False
    if os.path.exists(recipe_file):
        # Read the recipe file
        with open(recipe_file, 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            # Recover the arg from the recipe file
            recipe_arg = make_odm_args_from_recipe(data)
            if odm_args_checker(recipe_arg, args):
                print("Already processed. Try to copying to Processed folder.")
                if _process_outputs(args) == True:
                    return
                else:
                    print("Error in copying to Processed folder. Re-run ODM.")    
            else:
                # Reset the ODM if the arguments are different
                reset_odm_temp = True
    else:
        reset_odm_temp = True

    if reset_odm_temp:
        # Reset the ODM
        reset_odm(args)

    try:
        _create_directory_structure(args)

        image_pth = os.path.join(args.data_root_dir, 'Raw', args.year, args.experiment, args.location, args.population, args.date, args.platform, args.sensor, 'Images')
        # Check if the sensor is thermal
        if args.sensor.lower() == 'thermal':
            # Extract thermal images
            extracted_folder_name = 'Images_extracted'
            extract_thermal_images(image_pth, extracted_folder_name)
            image_pth = image_pth.replace('Images', extracted_folder_name)
        # Run ODM
        project_path = args.temp_dir

        if project_path[-1] == '/':
            project_path = project_path[:-1]
        if os.path.basename(project_path) != 'project':
            project_path = os.path.join(project_path, 'project')
        
        
        print('Project Path: ', project_path)

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
            # Create the command
            # It will mount pth to /datasets and image_pth to /datasets/code/images
            volumes = f"-v {project_path}:/datasets -v {image_pth}:/datasets/code/images -v /etc/timezone:/etc/timezone:ro -v /etc/localtime:/etc/localtime:ro"
            if check_nvidia_smi():
                docker_image = "--gpus all opendronemap/odm:gpu"
            else:
                docker_image = "opendronemap/odm"

            command = f"docker run --name GEMINI-Container -i --rm {volumes} {docker_image} --project-path /datasets code {options}" # 'code' is the default project name
            # Save settings to recipe yaml file
            with open(recipe_file, 'w') as file:
                # Export arg to data
                data = vars(args)
                data['image_pth'] = image_pth
                data['command'] = command
                yaml.dump(data, file, sort_keys=False)

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
    parser.add_argument('--date', type=str, help='Date of the data collection', default='2023-06-20')
    parser.add_argument('--platform', type=str, help='Platform used', default='Drone')
    parser.add_argument('--sensor', type=str, help='Sensor used', default='Thermal')
    parser.add_argument('--temp_dir', type=str, help='Temporary directory to store the images and gcp_list.txt',
                        default='/home/GEMINI/GEMINI-App-Data/temp/project')
    parser.add_argument('--data_root_dir', type=str, help='Root directory for the data', default='/home/GEMINI/GEMINI-App-Data')
    parser.add_argument('--reconstruction_quality', type=str, help='Reconstruction quality (high, low, custom)',
                        choices=[' high', 'low', 'custom'], default='low')
    parser.add_argument('--custom_options', nargs='+', help='Custom options for ODM (e.g. --orthophoto-resolution 0.01)', 
                        required=False)
    args = parser.parse_args()

    # Check if opendronemap is installed as a snap package
    if shutil.which('docker') is None:
        print("Error: docker is not installed. Please install docker")
        sys.exit(1)
    
    # Run ODM
    run_odm(args)