import os
from osgeo import gdal, osr, ogr
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import json
import argparse

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

def crop_xywh(src, x,y,w,h, rgb=False):
    # Get the image's starting coordinate
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
        data = np.transpose([band1.ReadAsArray(col1, row1, col2 - col1 + 1, row2 - row1 + 1),
                band2.ReadAsArray(col1, row1, col2 - col1 + 1, row2 - row1 + 1),
                band3.ReadAsArray(col1, row1, col2 - col1 + 1, row2 - row1 + 1)],(1,2,0))

    else:
        band = src.GetRasterBand(1)
        data = band.ReadAsArray(col1, row1, col2 - col1 + 1, row2 - row1 + 1)

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

def draw_geojson(dataset, total_height, plotlabels, fieldmap, mask_ds, rgb=False, debug = True):

    # get the first layer in the mask file
    mask_layer = mask_ds.GetLayer()

    # get the spatial reference of the mask layer
    mask_srs = mask_layer.GetSpatialRef()

    
    # get the input file's spatial reference
    input_srs = osr.SpatialReference()
    input_srs.ImportFromWkt(dataset.GetProjection())
    # create a transformation between the input and mask file spatial references
    transform = osr.CoordinateTransformation(mask_srs, input_srs) 

    # Draw on Map
    img = dataset_rgb.GetRasterBand(1)
    # Create an empty image
    height_map = np.ones((img.YSize,img.XSize,3),dtype=np.uint8)*255

    for feature, height in tqdm(zip(mask_layer, total_height)):
        # get the geometry of the feature
        geometry = feature.GetGeometryRef()
        # apply the transformation to the geometry
        geometry.Transform(transform)

        # get the extent of the feature in the input file's coordinate system
        minX, maxX, minY, maxY = geometry.GetEnvelope()

        # calculate the output file size and resolution
        pixel_width = dataset.GetGeoTransform()[1]
        pixel_height = dataset.GetGeoTransform()[5]
        x_res = int((maxX - minX) / pixel_width)
        y_res = int((maxY - minY) / -pixel_height)

        height_map = draw_rect_xywh(dataset, height_map, (255,0,0), maxX, maxY, maxX - minX, maxY - minY,rgb=rgb)
        
        
    return height_map

def calculate_exG_mask(clr_arr,threshold=0.9):
    # Calculate Ex
    clr_ratio = np.zeros((clr_arr.shape[0],clr_arr.shape[1],3),dtype=np.float32)
    clr_arr_float = clr_arr.astype(np.float32)
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

def crop_geojson(dataset, mask_ds, rgb=False, debug = False):

    # get the first layer in the mask file
    mask_layer = mask_ds.GetLayer()

    # get the spatial reference of the mask layer
    mask_srs = mask_layer.GetSpatialRef()

    # get the input file's spatial reference
    input_srs = osr.SpatialReference()
    input_srs.ImportFromWkt(dataset.GetProjection())
    # create a transformation between the input and mask file spatial references
    transform = osr.CoordinateTransformation(mask_srs, input_srs) 
    
    data_total = []
    for feature in tqdm(mask_layer):
        # get the geometry of the feature
        geometry = feature.GetGeometryRef()
        minLon, maxLon, minLat, maxLat = geometry.GetEnvelope()

        # apply the transformation to the geometry
        geometry.Transform(transform)

        # get the extent of the feature in the input file's coordinate system
        minX, maxX, minY, maxY = geometry.GetEnvelope()

        # calculate the output file size and resolution
        pixel_width = dataset.GetGeoTransform()[1]
        pixel_height = dataset.GetGeoTransform()[5]
        x_res = int((maxX - minX) / pixel_width)
        y_res = int((maxY - minY) / -pixel_height)

        cropped_img = crop_xywh(dataset, maxX, maxY, maxX - minX, maxY - minY,rgb=rgb)
        
        # Create a dict
        data_dict = {"Bed":feature.items()['Bed'],"Tier":feature.items()['Tier'],"img":cropped_img,
                     "minX":minX, "maxX":maxX, "minY":minY, "maxY":maxY,
                     "pixel_width":abs(pixel_width), "pixel_height":abs(pixel_height),
                     "minLon":minLon, "maxLon":maxLon, "minLat":minLat, "maxLat":maxLat}
        
        
        data_total.append(data_dict)

        if debug==True:        
            # Debug
            cv2.imwrite("data.jpg",cropped_img)
            break
        
        # if len(data_total) > 2:
        #     break
        
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

def process_tiff(tiff_files_rgb, tiff_files_dem, mask_file, output_file, crop=False, debug=False):

    # Check if the tiff_files is a list
    if type(tiff_files_rgb) == str:
        tiff_files_rgb = [tiff_files_rgb]

    # Check if the tiff_files is a list
    if type(tiff_files_dem) == str:
        tiff_files_dem = [tiff_files_dem]

    # open the mask file using ogr
    mask_ds = ogr.Open(mask_file)

    # get the first layer in the mask file
    mask_layer = mask_ds.GetLayer()

    # get the spatial reference of the mask layer
    mask_srs = mask_layer.GetSpatialRef()

    total_df = pd.DataFrame()

    # Load the GeoJSON file
    with open(mask_file) as f:
        json_fieldmap = json.load(f)



    for day, (tiff_dem,tiff_rgb) in enumerate(zip(tiff_files_dem,tiff_files_rgb)):
        # Load RGB
        print(f"Load rgb..Processing {tiff_rgb}")
        dataset_rgb = gdal.Open(tiff_rgb, gdal.GA_ReadOnly)
        data_rgb = crop_geojson(dataset_rgb, mask_ds, rgb=True)

        # Load Depth
        print(f"Load depth..Processing {tiff_dem}")
        dataset = gdal.Open(tiff_dem, gdal.GA_ReadOnly)
        data_depth = crop_geojson(dataset, mask_ds, rgb=False)

        # Create a crop dir if arg.crop is true
        if crop:
            if not os.path.exists("patches"):
                # Get the tiff file's dir
                save_dir = os.path.join(os.path.dirname(tiff_rgb),"patches")
                # Create a patches dir
                os.makedirs(save_dir,exist_ok=True)

        total_height = []
        total_vf = []
        print("Analyze...")
        Bed = []
        Tier = []
        total_lat = []
        total_lon = []

        plot_cordinates = []
        # TODO: Parallelize
        for rgb, depth in zip(tqdm(data_rgb), data_depth):
            # Extract
            rgb_img = rgb['img']
            depth_img = depth['img']
            rgb_resized = cv2.resize(rgb_img,dsize=(depth_img.shape[1],depth_img.shape[0]))
            mask = get_mask(rgb_resized)
            
            vegetation_fraction = np.sum(mask) / mask.size
            total_vf.append(vegetation_fraction)
            # Apply mask to image
            masked_img = cv2.bitwise_and(depth_img, depth_img, mask=mask)
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
                depth_map_normalized = cv2.normalize(masked_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                # create a grayscale image
                depth_map_gray = cv2.cvtColor(depth_map_normalized, cv2.COLOR_GRAY2BGR)
                cv2.imwrite("depth_map_gray.png",depth_map_gray)

            # Extract pixel values to list
            pixel_values = masked_img[np.where(masked_img != 0)].tolist()
            base = np.quantile(pixel_values,0.05)
            height = np.quantile(pixel_values,0.95)
            total_height.append(height - base)

            crop_height = height - base

            bed_no = rgb['Bed']
            tier_no = rgb['Tier']

            Bed.append(bed_no)
            Tier.append(tier_no)

            lon = (rgb['maxLon'] + rgb['minLon'])/2
            lat = (rgb['maxLat'] + rgb['minLat'])/2

            total_lon.append(lon)
            total_lat.append(lat)

            # Create a dict
            crop_location = {"Bed":bed_no,"Tier":tier_no,"maxX":rgb['maxX'], "maxY":rgb['maxY'], "minX":rgb['minX'], "minY":rgb['minY'],
                             "pixel_width":rgb['pixel_width'], "pixel_height":rgb['pixel_height'],
                             "minLon":rgb['minLon'], "maxLon":rgb['maxLon'], "minLat":rgb['minLat'], "maxLat":rgb['maxLat'],
                             }
            
            # Append to list
            plot_cordinates.append(crop_location)


            # Save image if arg.crop is true
            if crop:
               # Save RGB Image patch
               # use os.path.splitext() to split the filename and extension
               name_only, extension = os.path.splitext(tiff_rgb.split('/')[-1])
               # Generate save_path
               save_path = os.path.join(save_dir,f"Bed{bed_no}_Tier{tier_no}.png")
               #cv2.imwrite(save_path,rgb_resized)
               cv2.imwrite(save_path,rgb_img)


            # Dry run for debugging
            if debug:
                if len(Bed) > 10:
                    break

        
        df = pd.DataFrame({"Bed":Bed,
                           "Tier":Tier,
                           "height_95p_meters":total_height,
                           "vegetation_fraciton":total_vf,
                           "lon":total_lon,
                           "lat":total_lat, 
                           })
        
        for i in range(len(df)):
        
            feature = json_fieldmap['features'][i]

            # Check if bed and tier matches
            if feature['properties']['Bed'] != df.iloc[i]['Bed'] or feature['properties']['Tier'] != df.iloc[i]['Tier']:
                print("Bed and Tier does not match")
                continue
            

            feature['properties']['Height_95p_meters'] = round(df.iloc[i]['height_95p_meters'], 4)
            feature['properties']['Vegetation_Fraction'] = round(df.iloc[i]['vegetation_fraciton'], 4)

            json_fieldmap['features'][i] = feature

        # Write JSON file with new data
        with open(output_file, 'w') as f:
            f.write(json.dumps(json_fieldmap))



        




# Debug
if __name__ == "__main__":
    process_tiff(tiff_files_rgb="/home/GEMINI/GEMINI-Data/Processed/Davis/Legumes/2022-07-25/Drone/2022-07-25-P4-RGB.tif",
                 tiff_files_dem="/home/GEMINI/GEMINI-Data/Processed/Davis/Legumes/2022-07-25/Drone/2022-07-25-P4-DEM.tif",
                 mask_file="/home/GEMINI/GEMINI-Data/Processed/Davis/Legumes/Plot-Attributes-WGS84.geojson",
                 output_file="/home/GEMINI/GEMINI-Data/Processed/Davis/Legumes/2022-07-25/Results/2022-07-25-Drone-Traits-WGS84.geojson",
                 debug=True)