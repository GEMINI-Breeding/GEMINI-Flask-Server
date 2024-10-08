import os
from osgeo import gdal, osr, ogr
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import json
import argparse
import geopandas as gpd
import rasterio
import time
import concurrent.futures
from functools import partial

# Add the path to the scripts folder
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
import shared_states

DEBUG_ITER = 100

def find_drone_tiffs(image_folder:str) -> [str, str]:
    # List the files in the image folder
    files = os.listdir(image_folder)
    
    files = [file for file in files if file.endswith(".tif") and not "Pyramid" in file]

    # Find the RGB tiff file
    rgb_tiff_file = None
    for file in files:
        if "RGB" in file and "FLIR" not in file:
            rgb_tiff_file = os.path.join(image_folder, file)
            break

    if rgb_tiff_file is None:
        raise Exception("No RGB tiff file found in folder.")
    else:
        print(f"Found RGB tiff file: {rgb_tiff_file}")

    # Find the DEM tif file
    dem_tiff_file = None
    for file in files:
        if "DEM" in file:
            dem_tiff_file = os.path.join(image_folder, file)
            break
    
    if dem_tiff_file is None:
        #raise Exception("No DEM tiff file found in folder.")
        print("No DEM tiff file found in folder.")
        pass
    else:
        print(f"Found DEM tiff file: {dem_tiff_file}")

    # Find ther Thermal tif file
    thermal_tiff_file = None
    for file in files:
        if "FLIR" in file:
            thermal_tiff_file = os.path.join(image_folder, file)
            break

    if thermal_tiff_file is None:
        print("No Thermal tiff file found in folder.")
        pass
    else:
        print(f"Found Thermal tiff file: {thermal_tiff_file}")
        

    return rgb_tiff_file, dem_tiff_file, thermal_tiff_file

# Open tiff file using rasterio
def open_tiff_rasterio(tiff_file):
    # Open tiff file
    src = rasterio.open(tiff_file)
    return src

def get_img_data(src, rgb=False):
   
    if rgb == True:
        if 0:
            band1 = src.GetRasterBand(1)
            band2 = src.GetRasterBand(2)
            band3 = src.GetRasterBand(3)
        else:
            # Opencv BGR
            band1 = src.GetRasterBand(3)
            band2 = src.GetRasterBand(2)
            band3 = src.GetRasterBand(1)
        data = np.transpose([band1.ReadAsArray(),
                band2.ReadAsArray(),
                band3.ReadAsArray()],(1,2,0))

    else:
        band = src.GetRasterBand(1)
        data = band.ReadAsArray()

    return data    

def crop_xywh(src, x, y, w, h, image_type='rgb'):
    # Get the image's starting coordinate
    geotransform = src.GetGeoTransform()
    xinit = geotransform[0]
    yinit = geotransform[3]
    xsize = geotransform[1]
    ysize = geotransform[5]

    # p1 = point upper left of bounding box
    # p2 = point bottom right of bounding box
    p1 = (x, y)
    p2 = (x + w, y - h)

    row1 = int((p1[1] - yinit) / ysize)
    col1 = int((p1[0] - xinit) / xsize)

    row2 = int((p2[1] - yinit) / ysize)
    col2 = int((p2[0] - xinit) / xsize)

    # Get the dimensions of the array
    cols = src.RasterXSize
    rows = src.RasterYSize

    # Ensure the bounds are within the dimensions
    col1 = max(0, min(col1, cols - 1))
    row1 = max(0, min(row1, rows - 1))
    col2 = max(0, min(col2, cols - 1))
    row2 = max(0, min(row2, rows - 1))

    if col1 > col2 or row1 > row2:
        raise ValueError("Invalid crop bounds")

    if image_type == 'rgb':
        # OpenCV BGR
        band1 = src.GetRasterBand(3)
        band2 = src.GetRasterBand(2)
        band3 = src.GetRasterBand(1)
        data = np.transpose([band1.ReadAsArray(col1, row1, col2 - col1 + 1, row2 - row1 + 1),
                             band2.ReadAsArray(col1, row1, col2 - col1 + 1, row2 - row1 + 1),
                             band3.ReadAsArray(col1, row1, col2 - col1 + 1, row2 - row1 + 1)], (1, 2, 0))
    elif image_type == 'thermal':
        # Read last channel
        band = src.GetRasterBand(4)
        data = band.ReadAsArray(col1, row1, col2 - col1 + 1, row2 - row1 + 1)
        data = data / 100 - 273.15
    elif image_type == 'dem':
        # Read single channel
        band = src.GetRasterBand(1)
        data = band.ReadAsArray(col1, row1, col2 - col1 + 1, row2 - row1 + 1)
    else:
        # Raise error
        raise ValueError("image_type must be either rgb, thermal or dem")

    return data

def draw_rect_xywh(src, canvas, color, x,y,w,h, rgb=False):

    geotransform = src.GetGeoTransform()
    xinit = geotransform[0]
    yinit = geotransform[3]
    xsize = geotransform[1]
    ysize = geotransform[5]

    #p1 = point upper left of bounding box
    #p2 = point bottom right of bounding box
    p1 = (x, y) #(6, 5)
    p2 = (x+w, y-h) #(12, 14)

    row1 = int((p1[1] - yinit)/ysize)
    col1 = int((p1[0] - xinit)/xsize)

    row2 = int((p2[1] - yinit)/ysize)
    col2 = int((p2[0] - xinit)/xsize)

    cv2.rectangle(canvas, (row1, col1), (row2, col2), color, thickness=-1)

    return canvas


def calculate_exG_mask(clr_arr,threshold=0.5):
    # Calculate ExG
    clr_ratio = np.zeros((clr_arr.shape[0],clr_arr.shape[1],3),dtype=np.float32)
    clr_arr_float = clr_arr.astype(np.float32)

    # Calculate ratio
    for i in range(3):
        clr_ratio[:,:,i] = clr_arr_float[:,:,i] / (clr_arr_float[:,:,0] + clr_arr_float[:,:,1] + clr_arr_float[:,:,2])
    
    # Reset nan to 0
    clr_ratio[np.isnan(clr_ratio)] = 0

    # Calculate ExG
    exg = 2*clr_ratio[:,:,1] - clr_ratio[:,:,0] - clr_ratio[:,:,2]

    # Normalize
    exg = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # Debug
    if 0:
        cv2.imwrite("exg.png",exg)
    #TODO: Test triangle method
    if 1:
        # testing all thresholds from 0 to the maximum of the image
        threshold_range = range(np.max(exg)+1)
        criterias = [compute_otsu_criteria(exg, th) for th in threshold_range]

        # best threshold is the one minimizing the Otsu criteria
        best_threshold = threshold_range[np.argmin(criterias)]
    else:
        # Calulate threshold
        best_threshold = np.quantile(exg,threshold)
    
    # Apply mask
    mask = exg > best_threshold

    # Remove small objects
    # mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2,2),np.uint8))

    # Fill the holes
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    

    return np.array(mask,dtype=np.uint8)

def get_spatial_references(dataset, mask_ds):
    mask_layer = mask_ds.GetLayer()
    mask_srs = mask_layer.GetSpatialRef()

    input_srs = osr.SpatialReference()
    input_srs.ImportFromWkt(dataset.GetProjection())

    transform = osr.CoordinateTransformation(mask_srs, input_srs)

    return mask_layer, transform

def get_feature_geometry(feature, transform):
    geometry = feature.GetGeometryRef()
    minLon, maxLon, minLat, maxLat = geometry.GetEnvelope()

    geometry.Transform(transform)

    return geometry.GetEnvelope(), (minLon, maxLon, minLat, maxLat)

def get_image_resolution(dataset):
    pixel_width = dataset.GetGeoTransform()[1]
    pixel_height = dataset.GetGeoTransform()[5]

    return pixel_width, pixel_height

def crop_geojson(dataset, mask_ds, image_type='rgb', plots = None, debug = False):
    if type(mask_ds) == list:
        geojson_data = {
                        'type': 'FeatureCollection',
                        'features': []
                        }
        for feature in mask_ds:
            geojson_data['features'].append(feature)

        # Set the geojson data crs to WGS 84
        geojson_data['crs'] = {
            'type': 'name',
            'properties': {
                'name': 'EPSG:4326'
            }
        }
        # Save the geojson data to a file
        with open('temp.geojson', 'w') as f:
            json.dump(geojson_data, f)
        mask_ds = ogr.Open('temp.geojson')
        # Remove the temporary file
        os.remove('temp.geojson')


    mask_layer, transform = get_spatial_references(dataset, mask_ds)
    pixel_width, pixel_height = get_image_resolution(dataset)

    data_total = []
    for feature in tqdm(mask_layer):
        if plots:
            if feature.items()['Plot'] not in plots:
                continue
        (minX, maxX, minY, maxY), (minLon, maxLon, minLat, maxLat) = get_feature_geometry(feature, transform)

        cropped_img = crop_xywh(dataset, maxX, maxY, maxX - minX, maxY - minY, image_type=image_type)

        # Check if bed key exists
        if 'Bed' in feature.items():
            bed = feature.items()['Bed']
        elif 'column' in feature.items():
            bed = feature.items()['column']
        else:
            bed = None

        # Check if tier key exists
        if 'Tier' in feature.items():
            tier = feature.items()['Tier']
        elif 'row' in feature.items():
            tier = feature.items()['row']
        else:
            tier = None

        # Check if Plot key exists
        if 'Plot' in feature.items():
            plot = feature.items()['Plot']
        elif 'plot' in feature.items():
            plot = feature.items()['plot']
        else:
            plot = None

        # Check if Label key exists
        if 'Label' in feature.items():
            label = feature.items()['Label']
        elif 'accession' in feature.items():
            label = feature.items()['accession']

        data_dict = {
            "Bed": bed,
            "Tier": tier,
            "Plot": plot,
            "Label": label,
            "img": cropped_img,
            "minX": minX, "maxX": maxX, "minY": minY, "maxY": maxY,
            "pixel_width": abs(pixel_width), "pixel_height": abs(pixel_height),
            "minLon": minLon, "maxLon": maxLon, "minLat": minLat, "maxLat": maxLat
        }

        data_total.append(data_dict)

        if debug:
            cv2.imwrite("data.jpg", cropped_img)

    return data_total


def compute_otsu_criteria(im, th):
    """Otsu's method to compute criteria."""
    # create the thresholded image
    thresholded_im = np.zeros(im.shape)
    thresholded_im[im >= th] = 1

    # compute weights
    nb_pixels = im.size
    nb_pixels1 = np.count_nonzero(thresholded_im)
    weight1 = nb_pixels1 / nb_pixels
    weight0 = 1 - weight1

    # if one of the classes is empty, eg all pixels are below or above the threshold, that threshold will not be considered
    # in the search for the best threshold
    if weight1 == 0 or weight0 == 0:
        return np.inf

    # find all pixels belonging to each class
    val_pixels1 = im[thresholded_im == 1]
    val_pixels0 = im[thresholded_im == 0]

    # compute variance of these classes
    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0

    return weight0 * var0 + weight1 * var1

# Make a modular function
def get_mask(clr_arr):
    if 0:
        hsv_arr = cv2.cvtColor(clr_arr,cv2.COLOR_BGR2HSV)
        # hsv_arr = skimage.color.rgb2hsv(clr_arr)

        mask_h = hsv_arr[:,:,0] > 40
        mask_s = hsv_arr[:,:,1] > 15
        mask_v = hsv_arr[:,:,2] > 15
        return np.array(mask_h & mask_s & mask_v,dtype=np.uint8)
    else:
        # Apply mask
        mask = calculate_exG_mask(clr_arr)
        return np.array(mask,dtype=np.uint8)

def process_image(img_idx, data_rgb, data_depth, data_thermal, 
                  tiff_dem, tiff_thermal, save_cropped_imgs, save_dir, 
                  total_vf, total_height, total_temperature, 
                  Bed, Tier, total_lon, total_lat, debug):
    
    rgb = data_rgb[img_idx]
    depth = data_depth[img_idx]
    thermal = data_thermal[img_idx]

    # Extract
    rgb_img = rgb['img']
    depth_img = depth['img']
    rgb_resized = cv2.resize(rgb_img,dsize=(depth_img.shape[1],depth_img.shape[0]))
    mask = get_mask(rgb_resized)
    
    Vegetation_Fraction = np.sum(mask) / mask.size
    Vegetation_Fraction = round(Vegetation_Fraction, 4)
    total_vf.append(Vegetation_Fraction)

    if tiff_dem:
        # Apply mask to image
        masked_depth_img = cv2.bitwise_and(depth_img, depth_img, mask=mask)
        # Height Analyis
        # Debug
        if debug:
            cv2.imwrite("rgb_img.png", rgb_img)
            cv2.imwrite("mask.png", mask*255)

            # normalize depth values to range between 0 and 255
            depth_map_normalized = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            # create a grayscale image
            depth_map_gray = cv2.cvtColor(depth_map_normalized, cv2.COLOR_GRAY2BGR)
            cv2.imwrite("depth_map_gray_orig.png",depth_map_gray)

            # normalize depth values to range between 0 and 255
            depth_map_normalized = cv2.normalize(masked_depth_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            # create a grayscale image
            depth_map_gray = cv2.cvtColor(depth_map_normalized, cv2.COLOR_GRAY2BGR)
            cv2.imwrite("depth_map_gray.png",depth_map_gray)

        # Extract pixel values to list
        depth_pixel_values = masked_depth_img[np.where(masked_depth_img != 0)].tolist()
        base = np.quantile(depth_pixel_values,0.05)
        height = np.quantile(depth_pixel_values,0.95)
        crop_height = height - base
        crop_height = round(crop_height, 4)
        total_height.append(crop_height)
    else:
        print("No depth file found. Skipping depth analysis.")

    if tiff_thermal:
        # Thermal Analysis
        thermal_img = thermal['img']
        # Resize mask to thermal image size
        thermal_mask = cv2.resize(mask,dsize=(thermal_img.shape[1],thermal_img.shape[0]))
        masked_thermal_img = cv2.bitwise_and(thermal_img, thermal_img, mask=thermal_mask)
        thermal_pixel_values = masked_thermal_img[np.where(masked_thermal_img != 0)].tolist()
        avg_temp = np.mean(thermal_pixel_values)
        avg_temp = round(avg_temp, 2)
        total_temperature.append(avg_temp)
    else:
        print("No thermal file found. Skipping thermal analysis.")


    bed_no = rgb['Bed']
    tier_no = rgb['Tier']
    plot_no = rgb['Plot']

    Bed.append(bed_no)
    Tier.append(tier_no)

    lon = (rgb['maxLon'] + rgb['minLon'])/2
    lat = (rgb['maxLat'] + rgb['minLat'])/2

    total_lon.append(lon)
    total_lat.append(lat)


    # Save image if arg.crop is true
    if save_cropped_imgs:
        # Generate save_path
        save_path = os.path.join(save_dir,f"Plot_{plot_no}_Bed{bed_no}_Tier{tier_no}.png")
        #cv2.imwrite(save_path,rgb_resized)
        cv2.imwrite(save_path,rgb_img)


def parallel_process_images(data_rgb, data_depth, data_thermal, tiff_dem, tiff_thermal, 
                            save_cropped_imgs, save_dir, total_vf, total_height, total_temperature, 
                            Bed, Tier, total_lon, total_lat, debug):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Use partial to create a function with fixed arguments
        partial_process_image = partial(process_image, data_rgb=data_rgb, data_depth=data_depth, data_thermal=data_thermal, 
                                        tiff_dem=tiff_dem, tiff_thermal=tiff_thermal, save_cropped_imgs=save_cropped_imgs, save_dir=save_dir,
                                        total_vf=total_vf, total_height=total_height, total_temperature=total_temperature, 
                                        Bed=Bed, Tier=Tier, total_lon=total_lon, total_lat=total_lat, debug=debug)
        
        # Process images in parallel
        executor.map(partial_process_image, range(len(data_rgb)))

loaded_gdal_dict = {
    "tiff_files_rgb": None,
    "tiff_files_dem": None,
    "tiff_files_thermal": None,
    "gdal_dataset_rgb": None,
    "gdal_dataset_dem": None,
    "gdal_dataset_thermal": None,
    "data_rgb": None,
    "data_depth": None,
    "data_thermal": None
}

def process_tiff(tiff_files_rgb, tiff_files_dem, tiff_files_thermal, plot_geojson, output_geojson, save_cropped_imgs=False, debug=False):
    global loaded_gdal_dict

    # tiff_files_rgb is the essential file
    if type(tiff_files_rgb) == str:
        tiff_files_rgb = [tiff_files_rgb]

    # Check if the tiff_files is a list
    if type(tiff_files_dem) == str:
        tiff_files_dem = [tiff_files_dem]

    # Check if the tiff_files is a list
    if type(tiff_files_thermal) == str:
        tiff_files_thermal = [tiff_files_thermal]

    # open the mask file using ogr
    mask_ds = ogr.Open(plot_geojson)

    # open_tiff_rasterio(tiff_files_rgb[0])

    # Measure operation time if load the entire image
    if 0:
        start_time = time.time()
        dataset_rgb = gdal.Open(tiff_files_rgb[0], gdal.GA_ReadOnly)
        get_img_data(dataset_rgb, rgb=True)
        print("--- %s seconds ---" % (time.time() - start_time))


    # Load the GeoJSON file
    prog_started = False
    with open(plot_geojson) as f:
        json_fieldmap = json.load(f)

    for day in range(len(tiff_files_rgb)):
        start_time = time.time()
        tiff_rgb = tiff_files_rgb[day]
        
        output_path = os.path.dirname(tiff_rgb)
        with open(f"{output_path}/progress.txt", "w") as f:
            if not prog_started:
                f.write("0")
            
        # Load RGB
        print(f"Load rgb... {tiff_rgb}",flush=True)
        dataset_rgb = gdal.Open(tiff_rgb, gdal.GA_ReadOnly)
        data_rgb = crop_geojson(dataset_rgb, mask_ds, image_type='rgb', debug=debug)
        loaded_gdal_dict["data_rgb"] = data_rgb
        try:
            tiff_dem = tiff_files_dem[day]
        except:
            tiff_dem = None

        if tiff_dem:
            # Load Depth
            print(f"Load depth... {tiff_dem}",flush=True)
            dataset = gdal.Open(tiff_dem, gdal.GA_ReadOnly)
            data_depth = crop_geojson(dataset, mask_ds, image_type='dem', debug=debug)
            loaded_gdal_dict["data_depth"] = data_depth

        try:
            tiff_thermal = tiff_files_thermal[day]
        except:
            tiff_thermal = None

        if tiff_thermal:
            # Load Thermal
            print(f"Load thermal... {tiff_thermal}",flush=True)
            dataset_thermal = gdal.Open(tiff_thermal, gdal.GA_ReadOnly)
            data_thermal = crop_geojson(dataset_thermal, mask_ds, image_type='thermal', debug=debug)
            loaded_gdal_dict["data_thermal"] = data_thermal
        else:
            print("No thermal file found. Skipping thermal analysis.")
        
        print("Load images --- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()

        # Create a crop dir if arg.crop is true
        if save_cropped_imgs:
            if not os.path.exists("patches"):
                # Get the tiff file's dir
                save_dir = os.path.join(os.path.dirname(tiff_rgb),"patches")
                # Create a patches dir
                os.makedirs(save_dir,exist_ok=True)

        total_height = []
        total_vf = []
        total_temperature = []
        print("Analyze...")
        Bed = []
        Tier = []
        total_lat = []
        total_lon = []
            
        if 1:
            # TODO: Parallelize
            prog_started = True
            for img_idx in tqdm(range(len(data_rgb))):
                
                # check if stop signal is True
                if shared_states.stop_signal:
                    break
                
                with open(f"{output_path}/progress.txt", "w") as f:
                    if img_idx >= len(data_rgb)-(0.05*len(data_rgb)):
                        f.write("95")
                    else:
                        f.write(str((img_idx/len(data_rgb))*100))
                                
                rgb = data_rgb[img_idx]
                depth = data_depth[img_idx]

                # Extract
                rgb_img = rgb['img']
                depth_img = depth['img']
                rgb_resized = cv2.resize(rgb_img,dsize=(depth_img.shape[1],depth_img.shape[0]))
                mask = get_mask(rgb_resized)
                
                Vegetation_Fraction = round(np.sum(mask) / mask.size,4)
                total_vf.append(Vegetation_Fraction)

                if tiff_dem:
                    # Apply mask to image
                    masked_depth_img = cv2.bitwise_and(depth_img, depth_img, mask=mask)
                    # Height Analyis
                    # Debug
                    if debug:
                        cv2.imwrite("rgb_img.png", rgb_img)
                        cv2.imwrite("mask.png", mask*255)

                        # normalize depth values to range between 0 and 255
                        depth_map_normalized = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                        # create a grayscale image
                        depth_map_gray = cv2.cvtColor(depth_map_normalized, cv2.COLOR_GRAY2BGR)
                        cv2.imwrite("depth_map_gray_orig.png",depth_map_gray)

                        # normalize depth values to range between 0 and 255
                        depth_map_normalized = cv2.normalize(masked_depth_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                        # create a grayscale image
                        depth_map_gray = cv2.cvtColor(depth_map_normalized, cv2.COLOR_GRAY2BGR)
                        cv2.imwrite("depth_map_gray.png",depth_map_gray)

                    # Extract pixel values to list
                    depth_pixel_values = masked_depth_img[np.where(masked_depth_img != 0)].tolist()
                    if depth_pixel_values:
                        base = np.quantile(depth_pixel_values, 0.05)
                        height = np.quantile(depth_pixel_values, 0.95)
                        crop_height = round(height - base, 4)
                        total_height.append(crop_height)
                    else:
                        total_height.append(0)

                if tiff_thermal:
                    # Thermal Analysis
                    thermal = data_thermal[img_idx]
                    thermal_img = thermal['img']
                    # Resize mask to thermal image size
                    thermal_mask = cv2.resize(mask,dsize=(thermal_img.shape[1],thermal_img.shape[0]))
                    masked_thermal_img = cv2.bitwise_and(thermal_img, thermal_img, mask=thermal_mask)
                    thermal_pixel_values = masked_thermal_img[np.where(masked_thermal_img != 0)].tolist()
                    avg_temp = round(np.mean(thermal_pixel_values),2)
                    total_temperature.append(avg_temp)


                bed_no = rgb['Bed']
                tier_no = rgb['Tier']
                plot_no = rgb['Plot']

                Bed.append(bed_no)
                Tier.append(tier_no)

                lon = (rgb['maxLon'] + rgb['minLon'])/2
                lat = (rgb['maxLat'] + rgb['minLat'])/2

                total_lon.append(lon)
                total_lat.append(lat)


                # Save image if arg.crop is true
                if save_cropped_imgs:
                    # Save RGB Image patch
                    # use os.path.splitext() to split the filename and extension
                    name_only, extension = os.path.splitext(tiff_rgb.split('/')[-1])
                    
                    # Generate save_path
                    save_path = os.path.join(save_dir,f"Plot_{plot_no}_Bed{bed_no}_Tier{tier_no}.png")
                    
                    #cv2.imwrite(save_path,rgb_resized)
                    cv2.imwrite(save_path,rgb_img)


                # Dry run for debugging
                if debug:
                    if len(Bed) > DEBUG_ITER:
                        break
        else:
            # Call the parallelized function
            # Create a crop dir if arg.crop is true
            save_dir = os.path.join(os.path.dirname(tiff_rgb),"patches")
            parallel_process_images(data_rgb, data_depth, data_thermal, tiff_dem, 
                                    tiff_thermal, save_cropped_imgs, save_dir, total_vf, 
                                    total_height, total_temperature, Bed, Tier, total_lon, total_lat, debug)

        # check if stop signal is True
        if shared_states.stop_signal:
            break
        print("Analyze images --- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        
        df = pd.DataFrame({"Bed":Bed,
                           "Tier":Tier,
                           "lon":total_lon,
                           "lat":total_lat, 
                           })
        
        if len(total_height) > 0:
            df["Height_95p_meters"] = total_height
        
        if len(total_vf) > 0:
            df["Vegetation_Fraction"] = total_vf
        
        if len(total_temperature) > 0:
            df["Avg_Temp_C"] = total_temperature

        if 0:
            for i in range(len(df)):
            
                feature = json_fieldmap['features'][i]

                # Check if bed and tier matches
                if feature['properties']['Bed'] != df.iloc[i]['Bed'] or feature['properties']['Tier'] != df.iloc[i]['Tier']:
                    print("Bed and Tier does not match")
                    continue
                
                feature['properties']['Height_95p_meters'] = round(df.iloc[i]['Height_95p_meters'], 4)
                feature['properties']['Vegetation_Fraction'] = round(df.iloc[i]['Vegetation_Fraction'], 4)
                feature['properties']['Avg_Temp_C'] = round(df.iloc[i]['Avg_Temp_C'], 4)

                json_fieldmap['features'][i] = feature

            # Write JSON file with new data
            gpd.GeoDataFrame.from_features(json_fieldmap).to_file(output_geojson, driver='GeoJSON')
        else:
            # Merge the two dataframes and save to geojson
            # print(df)
            # Create Bed if not exists
            for feature in json_fieldmap['features']:
                if 'Bed' not in feature['properties']:
                    feature['properties']['Bed'] = feature['properties']['column']
                    feature['properties']['Tier'] = feature['properties']['row']
            # gpd.GeoDataFrame.merge(gpd.GeoDataFrame.from_features(json_fieldmap), df, on=['Bed','Tier']).to_file(output_geojson, driver='GeoJSON')
            gdf = gpd.GeoDataFrame.from_features(json_fieldmap)
            merged_gdf = gdf.merge(df, on=['Bed', 'Tier'])
            merged_gdf = merged_gdf.fillna("None")
            merged_gdf = merged_gdf.drop(columns=['col', 'row', 'column'], errors='ignore')
            merged_gdf.to_file(output_geojson, driver='GeoJSON')
    
    # check if stop signal is True
    if shared_states.stop_signal:
        return False
    
    with open(f"{output_path}/progress.txt", "w") as f:
                f.write("100")
    print("Done.")
    return True


def query_drone_images(args_dict, data_root_dir):
    global loaded_gdal_dict
    geojson_features = args_dict['geoJSON']
    year = args_dict['selectedYearGCP']
    experiment = args_dict['selectedExperimentGCP']
    location = args_dict['selectedLocationGCP']
    population = args_dict['selectedPopulationGCP']
    date = args_dict['selectedDateQuery']
    sensor = args_dict['selectedSensorQuery']
    platform = args_dict['selectedPlatformQuery']
    plots = args_dict['selectedPlots']

    # Construct the CSV path from the state variables
    tiff_path = os.path.join(data_root_dir, 'Processed', year, experiment, location, population, date, platform, sensor)

    # Find tiff file
    tiff_files_rgb, tiff_files_dem, tiff_files_thermal = find_drone_tiffs(tiff_path)

    # Load tiff files if not already loaded
    if tiff_files_rgb:
        if tiff_files_rgb != loaded_gdal_dict["tiff_files_rgb"]:
            loaded_gdal_dict["tiff_files_rgb"] = tiff_files_rgb
            loaded_gdal_dict["gdal_dataset_rgb"] = gdal.Open(tiff_files_rgb, gdal.GA_ReadOnly)
    if tiff_files_dem:
        if tiff_files_dem != loaded_gdal_dict["tiff_files_dem"]:
            loaded_gdal_dict["tiff_files_dem"] = tiff_files_dem
            loaded_gdal_dict["gdal_dataset_dem"] = gdal.Open(tiff_files_dem, gdal.GA_ReadOnly)
    if tiff_files_thermal:
        if tiff_files_thermal != loaded_gdal_dict["tiff_files_thermal"]:
            loaded_gdal_dict["tiff_files_thermal"] = tiff_files_thermal
            loaded_gdal_dict["gdal_dataset_thermal"] = gdal.Open(tiff_files_thermal, gdal.GA_ReadOnly)

    # Correct geojson_features key "plot" to "Plot" for temproary compatibility
    for feature in geojson_features:
        if 'plot' in feature['properties']:
            feature['properties']['Plot'] = feature['properties']['plot']

        if 'Bed' not in feature['properties']:
            feature['properties']['Bed'] = feature['properties']['column']
            feature['properties']['Tier'] = feature['properties']['row']

        if 'Label' not in feature['properties']:
            feature['properties']['Label'] = feature['properties']['accession']

    # Crop the images based on the geojson features
    data_rgb = crop_geojson(loaded_gdal_dict["gdal_dataset_rgb"], geojson_features, image_type='rgb', plots=plots, debug=False)

    # Save imges to a folder to be used in the frontend
    save_dir = os.path.join(tiff_path, 'cropped_images')
    os.makedirs(save_dir, exist_ok=True)

    filtered_images = []
    filtered_labels = []
    filtered_plots = []
    for i, data_line in enumerate(data_rgb):
        plot_id = data_line['Plot']
        # Resize image to 1080 height
        desired_height = 640
        img_resized = cv2.resize(data_line['img'], (int(data_line['img'].shape[1] * desired_height / data_line['img'].shape[0]), desired_height))
        cv2.imwrite(os.path.join(save_dir, f"plot_{plot_id}.png"), img_resized)
        # Calculate relative path to data_root_dir
        filtered_images_path = os.path.relpath(os.path.join(save_dir, f"plot_{plot_id}.png"), data_root_dir)
        filtered_images.append(filtered_images_path)
        filtered_labels.append(data_line['Label'])
        filtered_plots.append(plot_id)
        

    filtered_images = [{'imageName': image, 'label': label, 'plot': plot} for image, label, plot in zip(filtered_images, filtered_labels, filtered_plots)]

    # Sort the filtered_images by label
    filtered_images = sorted(filtered_images, key=lambda x: x['label'])

    return filtered_images

# Debug
if __name__ == "__main__":
    process_tiff(tiff_files_rgb="/home/GEMINI/Dataset_processing/Davis_Legumes/2022-09-05/Drone/metashape/2022-09-05-P4-RGB.tif",
                 tiff_files_dem="/home/GEMINI/Dataset_processing/Davis_Legumes/2022-09-05/Drone/metashape/2022-09-05-P4-DEM.tif",
                 tiff_files_thermal="",
                 plot_geojson="/home/lion397/GEMINI/GEMINI-App-Data/Intermediate/2022/GEMINI/Davis/Legumes/Plot-Boundary-WGS84.geojson",
                 output_geojson="/home/GEMINI/Dataset_processing/Davis_Legumes/2022-09-05/Drone/metashape/Traits-WGS84.geojson",
                 save_cropped_imgs=True,
                 debug=False)
