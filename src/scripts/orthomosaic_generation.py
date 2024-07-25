# Standard library imports
import os
import shutil
import subprocess
import sys
import argparse
from glob import glob
from math import log
from concurrent.futures import ThreadPoolExecutor

# Third-party library imports
import cv2
from numpy import std
from tqdm import tqdm
import yaml

# Local application/library specific imports
# Add current directory to path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from create_geotiff_pyramid import create_tiled_pyramid

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

 
    if 0:
        # Copy the images to the temporary directory
        image_pth = os.path.join(args.data_root_dir, 'Raw', args.year, args.experiment, args.location, args.population, args.date, args.platform, args.sensor, 'Images')
        print(f"Image Path: {image_pth}")
        os.makedirs(os.path.join(pth, 'code', 'images'))
        images = os.listdir(os.path.join(image_pth))
        extensions = ('.jpg', '.tif','.png')
        images = [x for x in images if x.lower().endswith(extensions)]
        if 0:
            # Copy the images to the temporary directory
            with ThreadPoolExecutor() as executor:
                executor.map(lambda im_name: _copy_image(image_pth, 
                                                    os.path.join(pth, 'code', 'images'), 
                                                    im_name), 
                                        images)
        else:
            print("Copying images")
            for im_name in tqdm(images):
                _copy_image(image_pth, os.path.join(pth, 'code', 'images'), im_name)
    else:
        # Do nothing because docker will mount the image folder to the container
        pass

    # Copy the gcp_list.txt to the temporary directory
    gcp_pth = os.path.join(args.data_root_dir, 'Intermediate', args.year, args.experiment, args.location, args.population, args.date, args.platform, args.sensor, 'gcp_list.txt')
    print(f"GCP Path: {gcp_pth}")
    # Check if the file has more than 1 lines
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
    
def make_odm_args_from_path(path):
    '''
    Generate an argparse.Namespace object from a path to the images.
    Example path string: Raw/2022/GEMINI/Davis/Legumes/2022-06-20/Drone/RGB/Images
    '''
    parts = path.split('/')
    sensor = parts[-2]
    platform = parts[-3]
    date = parts[-4]
    population = parts[-5]
    location = parts[-6]
    experiment = parts[-7]
    year = parts[-8]
    data_root_dir = parts[-9]
    temp_dir = os.path.join(data_root_dir, 'temp', 'project')

    return make_odm_args(data_root_dir, location, population, date, year, experiment, platform, sensor, temp_dir, 'Low', [])



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

def reset_odm(data_root_dir):
    # Delete existing folders
    temp_path = os.path.join(data_root_dir, 'temp')
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


def run_odm(args):
    '''
    Run ODM on the temporary directory.
    '''
    
    # Check if the already processed data is not in the Processed folder
    # Copy the temp to the Processed folder without deleting the temp, and finish
    # Check if the log file exists

    pth = args.temp_dir
    recipe_file = os.path.join(pth, 'code', 'recipe.yaml')
    reset_odm_temp = False
    if os.path.exists(recipe_file):
        # Read the recipe file
        with open(recipe_file, 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            image_pth = data['image_pth']
            prev_arg = make_odm_args_from_path(image_pth)
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
        # Reset the ODM
        reset_odm(args.data_root_dir)

    try:
        _create_directory_structure(args)

        # Run ODM
        pth = args.temp_dir

        if pth[-1] == '/':
            pth = pth[:-1]
        if os.path.basename(pth) != 'project':
            pth = os.path.join(pth, 'project')
        
        
        print('Project Path: ', pth)
        image_pth = os.path.join(args.data_root_dir, 'Raw', args.year, args.experiment, args.location, args.population, args.date, args.platform, args.sensor, 'Images')
        options = ""
        log_file = os.path.join(pth, 'code', 'logs.txt')
        with open(log_file, 'w') as f:
            # See options from https://docs.opendronemap.org/arguments/
            #common_options = "--dsm --orthophoto-resolution 2.0 --sfm-algorithm planar" # orthophoto-resolution gsd is usually 0.27cm/pixel
            common_options = "--dsm --orthophoto-resolution 2.0" # orthophoto-resolution gsd is usually 0.27cm/pixel
            if args.reconstruction_quality == 'Custom':
                #process = subprocess.Popen(['opendronemap', 'code', '--project-path', pth, *args.custom_options, '--dsm'], stdout=f, stderr=subprocess.STDOUT)
                options = f"{args.custom_options} {common_options}"
                print('Starting ODM with custom options...')
            elif args.reconstruction_quality == 'Low':
                options = f"--pc-quality medium --min-num-features 8000 {common_options}" 
                print('Starting ODM with low options...')
            elif args.reconstruction_quality == 'Lowest':
                options = f"--fast-orthophoto --dsm {common_options}"
                print('Starting ODM with Lowest options...')
            elif args.reconstruction_quality == 'High':
                options = f"--pc-quality high --min-num-features 16000 {common_options}"
                print('Starting ODM with high options...')
            else:
                raise ValueError('Invalid reconstruction quality: {}. Must be one of: low, high, custom'.format(args.reconstruction_quality))
            # Create the command
            # It will mount pth to /datasets and image_pth to /datasets/code/images
            volumes = f"-v {pth}:/datasets -v {image_pth}:/datasets/code/images -v /etc/timezone:/etc/timezone:ro -v /etc/localtime:/etc/localtime:ro"
            if check_nvidia_smi():
                docker_image = "--gpus all opendronemap/odm:gpu"
            else:
                docker_image = "opendronemap/odm"

            command = f"docker run -i --rm {volumes} {docker_image} --project-path /datasets code {options}" # 'code' is the default project name
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
            recipe_file = os.path.join(pth, 'code', 'recipe.yaml')
            with open(recipe_file, 'w') as file:
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
    parser = argparse.ArgumentParser(description='Generate an orthomosaic for a set of images')
    parser.add_argument('--date', type=str, help='Date of the data collection')
    parser.add_argument('--location', type=str, help='Location of the data collection')
    parser.add_argument('--population', type=str, help='Population for the dataset')
    parser.add_argument('--year', type=str, help='Year of the data collection')
    parser.add_argument('--experiment', type=str, help='Experiment name')
    parser.add_argument('--sensor', type=str, help='Sensor used')
    parser.add_argument('--temp_dir', type=str, help='Temporary directory to store the images and gcp_list.txt',
                        default='/home/GEMINI/temp/project')
    parser.add_argument('--data_root_dir', type=str, help='Root directory for the data', default='/home/GEMINI/GEMINI-Data')
    parser.add_argument('--reconstruction_quality', type=str, help='Reconstruction quality (high, low, custom)',
                        choices=['High', 'Low', 'Custom'], default='low')
    parser.add_argument('--custom_options', nargs='+', help='Custom options for ODM (e.g. --orthophoto-resolution 0.01)', 
                        required=False)
    args = parser.parse_args()

    # Check if opendronemap is installed as a snap package
    if shutil.which('opendronemap') is None:
        print("Error: opendronemap is not installed. Please install it using 'sudo snap install opendronemap --edge'")
        sys.exit(1)
    
    # Run ODM
    run_odm(args)