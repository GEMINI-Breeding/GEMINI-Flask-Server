import os
from glob import glob
from tqdm import tqdm
import cv2
import pandas as pd

# Add current script to the path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),"../"))

from thermal_camera.flir_image_extractor import FlirImageExtractor
from thermal_camera.process_flir import warp_image
from utils import dms_to_decimal

# def extract_thermal_images(image_pth, extracted_folder_name,show_progress=True):
#     '''
#     # Extract thermal images
#     # This function will extract RGB and thermal images from the FLIR images
#     # How ODM deal with thermal images: https://docs.opendronemap.org/thermal/
#     # How ODM deal with multispectral images: https://docs.opendronemap.org/multispectral/
#     '''

#     fie = FlirImageExtractor(exiftool_path="exiftool",use_calibration_data=True, is_debug=False)

#     # Create the folder
#     extracted_folder = os.path.join(image_pth, "../",extracted_folder_name)
#     if not os.path.exists(extracted_folder):
#         os.makedirs(extracted_folder)

#     # Create rgb and thermal folders
#     rgb_folder = os.path.join(extracted_folder, 'rgb')
#     thermal_folder = os.path.join(extracted_folder, 'thermal')
#     if not os.path.exists(rgb_folder):
#         os.makedirs(rgb_folder)
#     if not os.path.exists(thermal_folder):
#         os.makedirs(thermal_folder)

#     # Get the list of images
#     images = glob(os.path.join(image_pth, '*.jpg'))
#     images = sorted(images)
    
#     # Prepare geo.txt file
#     geo_txt_path = os.path.join(image_pth, "../", 'geo.txt')

#     # Create a empty pandas dataframe to store the metadata
#     metadata_df = pd.DataFrame()

#     # Check if the geo.txt file exists
#     if os.path.exists(geo_txt_path):
#         # Read the existing file
#         with open(geo_txt_path, "r") as f:
#             lines = f.readlines()
#             # Check if the file is empty
#             if len(lines) > 0:
#                 # Get the last line
#                 last_line = lines[-1]
#                 # Check if the last line is empty
#                 if last_line.strip() == "":
#                     # Remove the last line
#                     lines = lines[:-1]
#                 else:
#                     # Add a new line
#                     lines.append("\n")
#     else:   
#         # Create a new file
#         with open(geo_txt_path, "w") as f:
#             # Write the projection to the file
#             f.write("EPSG:4326\n")

#     # Extract the thermal images
#     print("Extracting thermal images...")
#     if show_progress:
#         images = tqdm(images)
    
    
#     for i, image in enumerate(images):
#         image_name = os.path.basename(image)
#         # Change the extension to .tiff
#         rgb_image_name = image_name.replace('.jpg', '_rgb.tif')
#         thermal_image_name = image_name.replace('.jpg', '_thermal.tif')
        
#         # Check if the image is already extracted
#         if os.path.exists(os.path.join(extracted_folder, rgb_folder, rgb_image_name)):
#             # Also check if the thermal image is already extracted
#             if os.path.exists(os.path.join(extracted_folder, thermal_folder, thermal_image_name)):
#                 continue

#         fie.process_image(image)
#         # Extract Thermal Image
#         raw_rgb = fie.get_rgb_np()
#         # The the raw thermal image
#         thermal_image_np = fie.get_RawThermalImage()
#         thermal_image_np = thermal_image_np.astype('uint16')
#         # Warp RGB
#         warped_rgb, _  = warp_image(raw_rgb, thermal_image_np, fie.meta, obj_distance=10.0)

#         if 1:
#             # BGR to RGB
#             warped_rgb = cv2.cvtColor(warped_rgb, cv2.COLOR_BGR2RGB)
#             # Save the RGB image
#             cv2.imwrite(os.path.join(extracted_folder, rgb_folder, rgb_image_name), warped_rgb)

#             # Save the thermal image
#             cv2.imwrite(os.path.join(extracted_folder, thermal_folder, thermal_image_name), thermal_image_np)
#         else:
#             # Convert into uint16
#             warped_rgb_uint16 = (warped_rgb * 2**8).astype('uint16')

#             # Make RGB-Thermal 4 channel image
#             rgb_thermal = cv2.merge([warped_rgb_uint16[:,:,0], warped_rgb_uint16[:,:,1], warped_rgb_uint16[:,:,2], thermal_image_np])

#             # Save the image
#             cv2.imwrite(os.path.join(extracted_folder, image_name), rgb_thermal)

#             # Write the calibration data to a file

#         # Add fie.meta to the metadata dataframe
#         new_row = pd.DataFrame([fie.meta])
#         metadata_df = pd.concat([metadata_df, new_row], ignore_index=True)

#         # Dry run
#         # if i > 10:
#         #     break

#         # Write the geo.txt file
#         gps_info = fie.extract_gps_info()
#         gps_info_keys = gps_info.keys()
#         # Check if the GPS info is available
#         if 'GPSLatitude' in gps_info_keys and 'GPSLongitude' in gps_info_keys and 'GPSAltitude' in gps_info_keys:
#             lat_DMS = gps_info['GPSLatitude']
#             lon_DMS = gps_info['GPSLongitude']
#             height_m = gps_info['GPSAltitude']
#             lat = dms_to_decimal(lat_DMS)
#             lon = dms_to_decimal(lon_DMS)
#             height = float(height_m.replace(' m', ''))
#             with open(geo_txt_path, "w") as f:
#                 f.write(f"{image_name} {lon} {lat} {height}\n")

             

#         # Save the metadata to a csv file
#         metadata_df.to_csv(os.path.join(extracted_folder, 'metadata.csv'), index=False)

#     print("Thermal images extracted.")


def extract_thermal_images_NIRFormat(image_pth, extracted_folder_name,show_progress=True):
    '''
    # Extract thermal images
    # This function will extract RGB and thermal images from the FLIR images
    '''

    fie = FlirImageExtractor(exiftool_path="exiftool",use_calibration_data=True, is_debug=False)

    # Create the folder
    extracted_folder = os.path.join(image_pth, "../",extracted_folder_name)
    if not os.path.exists(extracted_folder):
        os.makedirs(extracted_folder)

    # Get the list of images
    images = glob(os.path.join(image_pth, '*.jpg'))
    images = sorted(images)
    
    # Prepare geo.txt file
    geo_txt_path = os.path.join(image_pth, "../", 'geo.txt')

    # Create a empty pandas dataframe to store the metadata
    metadata_df = pd.DataFrame()

    # Check if the geo.txt file exists
    processed_img_names = []
    if os.path.exists(geo_txt_path):
        # Read the existing file
        with open(geo_txt_path, "r") as f:
            lines = f.readlines()
            # Check if the file is empty
            for line in lines:
                img_name = line.split()[0]
                if "EPSG" not in img_name:
                    processed_img_names.append(img_name)
    else:   
        # Create a new file
        with open(geo_txt_path, "w") as f:
            # Write the projection to the file
            f.write("EPSG:4326\n")

    # Extract the thermal images
    print("Extracting thermal images...")
    if show_progress:
        images = tqdm(images)
    
    for i, image in enumerate(images):
        image_name = os.path.basename(image)
        # Extract Image index from name. Example Name: 240715_IMG_03177.jpg
        image_index = image_name.split('_')[-1].split('.')[0]
        # # Change the extension to .tiff
        # rgb_image_name = f"{image_index}_RGB.jpg"
        # thermal_image_name = f"{image_index}_NIR.tif"
        
        rgb_image_name = f"DJI_{int(image_index):03d}0.JPG"
        thermal_image_name = f"DJI_{int(image_index):03d}1.TIF"
        
        # Check if the image is already extracted
        if os.path.exists(os.path.join(extracted_folder, rgb_image_name)) or rgb_image_name in processed_img_names:
            # Also check if the thermal image is already extracted
            if os.path.exists(os.path.join(extracted_folder, thermal_image_name)):
                continue

        fie.process_image(image)
        # Extract Thermal Image
        raw_rgb = fie.get_rgb_np()
        # The the raw thermal image
        thermal_image_np = fie.get_RawThermalImage()
        thermal_image_np = thermal_image_np.astype('uint16')
        # Warp RGB
        warped_rgb, _  = warp_image(raw_rgb, thermal_image_np, fie.meta, obj_distance=10.0)

        # BGR to RGB
        warped_rgb = cv2.cvtColor(warped_rgb, cv2.COLOR_BGR2RGB)
        # Save the RGB image
        cv2.imwrite(os.path.join(extracted_folder, rgb_image_name), warped_rgb)

        # Save the thermal image
        cv2.imwrite(os.path.join(extracted_folder, thermal_image_name), thermal_image_np)

        # TODO: Write the calibration data to a file

        # Add fie.meta to the metadata dataframe
        new_row = pd.DataFrame([fie.meta])
        metadata_df = pd.concat([metadata_df, new_row], ignore_index=True)

        # Dry run
        # if i > 10:
        #     break

        # Write the geo.txt file
        gps_info = fie.extract_gps_info()
        gps_info_keys = gps_info.keys()
        # Check if the GPS info is available
        if 'GPSLatitude' in gps_info_keys and 'GPSLongitude' in gps_info_keys and 'GPSAltitude' in gps_info_keys:
            lat_DMS = gps_info['GPSLatitude']
            lon_DMS = gps_info['GPSLongitude']
            height_m = gps_info['GPSAltitude']
            lat = dms_to_decimal(lat_DMS)
            lon = dms_to_decimal(lon_DMS)
            height = float(height_m.replace(' m', ''))
            with open(geo_txt_path, "a") as f:
                f.write(f"{rgb_image_name} {lon} {lat} {height}\n")
                f.write(f"{thermal_image_name} {lon} {lat} {height}\n")

        # Save the metadata to a csv file
        metadata_df.to_csv(os.path.join(extracted_folder, 'metadata.csv'), index=False)

    print("Thermal images extracted.")




if __name__ == "__main__":
    image_pth = "/mnt/d/GEMINI-App-Data-DEMO/Raw/2024/GEMINI/Davis/Legumes/2024-07-15/Drone/thermal/Images"
    extracted_folder_name = "Images_extracted"
    extract_thermal_images_NIRFormat(image_pth, extracted_folder_name)
