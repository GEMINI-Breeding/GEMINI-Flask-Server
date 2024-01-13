from glob import glob
from math import log
import os
import shutil
import subprocess
import sys
import cv2
import argparse
from concurrent.futures import ThreadPoolExecutor

from numpy import std

def _copy_image(src_folder, dest_folder, image_name):
    
    src_path = os.path.join(src_folder, image_name)
    dest_path = os.path.join(dest_folder, image_name)

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
        os.makedirs(os.path.join(pth, 'code', 'images'))

    # Copy the images to the temporary directory
    image_pth = os.path.join(args.data_root_dir, 'Raw', args.location, args.population, args.date, args.sensor, 'Images')
    images = os.listdir(os.path.join(image_pth))
    images = [x for x in images if x.endswith('.JPG') or x.endswith('.jpg') or x.endswith('.tif') or x.endswith('.TIF')]

    # Copy the images to the temporary directory
    with ThreadPoolExecutor() as executor:
        executor.map(lambda im_name: _copy_image(image_pth, 
                                            os.path.join(pth, 'code', 'images'), 
                                            im_name), 
                                images)

    # Copy the gcp_list.txt to the temporary directory
    gcp_pth = os.path.join(args.data_root_dir, 'Processed', args.location, args.population, args.date, args.sensor, 'gcp_list.txt')
    shutil.copy(gcp_pth, os.path.join(pth, 'code', 'gcp_list.txt'))

def _process_outputs(args):

    pth = args.temp_dir

    if pth[-1] == '/':
        pth = pth[:-1]
    if os.path.basename(pth) != 'project':
        pth = os.path.join(pth, 'project')    

    ortho_file = os.path.join(pth, 'code', 'odm_orthophoto', 'odm_orthophoto.tif')
    dem_file = os.path.join(pth, 'code', 'odm_dem', 'dsm.tif')

    output_folder = os.path.join(args.data_root_dir, 'Processed', args.location, args.population, args.date, args.sensor)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    shutil.copy(ortho_file, os.path.join(output_folder, args.date+'-'+'.tif'))
    shutil.copy(dem_file, os.path.join(output_folder, 'dem.tif'))

    # Move ODM outputs to /var/tmp so they can be accessed if needed but deleted eventually
    if not os.path.exists(os.path.join('/var/tmp', args.location, args.population, args.date, args.sensor)):
        os.makedirs(os.path.join('/var/tmp', args.location, args.population, args.date, args.sensor))
    shutil.move(os.path.join(pth, 'code'), os.path.join('/var/tmp', args.location, args.population, args.date, args.sensor))
    
def run_odm(args):
    '''
    Run ODM on the temporary directory.
    '''

    _create_directory_structure(args)

    # Run ODM
    pth = args.temp_dir

    if pth[-1] == '/':
        pth = pth[:-1]
    if os.path.basename(pth) != 'project':
        pth = os.path.join(pth, 'project')
    
    log_file = os.path.join(pth, 'code', 'logs.txt')

    with open(log_file, 'w') as f:
        if args.reconstruction_quality == 'Custom':
            process = subprocess.Popen(['opendronemap', 'code', '--project-path', pth, *args.custom_options], stdout=f, stderr=subprocess.STDOUT)
        elif args.reconstruction_quality == 'Low':
            process = subprocess.Popen(['opendronemap', 'code', '--project-path', pth, '--pc-quality', 'medium', '--min-num-features', '8000'], stdout=f, stderr=subprocess.STDOUT)
        elif args.reconstruction_quality == 'High':
            process = subprocess.Popen(['opendronemap', 'code', '--project-path', pth, '--pc-quality', 'high', '--min-num-features', '16000'], stdout=f, stderr=subprocess.STDOUT)
        else:
            raise ValueError('Invalid reconstruction quality: {}. Must be one of: low, high, custom'.format(args.reconstruction_quality))
        process.wait()
    
    _process_outputs(args)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate an orthomosaic for a set of images')
    parser.add_argument('--date', type=str, help='Date of the data collection')
    parser.add_argument('--location', type=str, help='Location of the data collection')
    parser.add_argument('--population', type=str, help='Population for the dataset')
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