# Standard imports
import os
import sys
import numpy as np
import cv2
from tqdm import tqdm
import multiprocessing as mp
from fractions import Fraction
# from read_depth_tiff import disp_depth
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider

# Custom imports
sys.path.append(os.path.dirname(os.path.abspath(__file__))) # Add path to the folder containing the current python file
from flir_image_extractor import FlirImageExtractor

def get_file_list(dataroot, subdir, exts):
    img_dir = os.path.join(dataroot, subdir)
    file_list = [os.path.join(img_dir, x) for x in os.listdir(img_dir) if x.split('.')[-1].lower() in exts and x[0] != "."]
    file_list.sort()
    return file_list

def change_to_rational(number):
    """convert a number to rantional
    Keyword arguments: number
    return: tuple like (1, 2), (numerator, denominator)
    """
    f = Fraction(str(number))
    return (f.numerator, f.denominator)

def generate_gps_time_string(year, month, day, hour, min, sec):
    gpsTimeString = f"{year}:{month}:{day} {hour}:{min}:{sec}"

    return gpsTimeString

def to_deg(value, loc):
    """convert decimal coordinates into degrees, munutes and seconds tuple
    Keyword arguments: value is float gps-value, loc is direction list ["S", "N"] or ["W", "E"]
    return: tuple like (25, 13, 48.343 ,'N')
    """
    if value < 0:
      loc_value = loc[0]
    elif value > 0:
      loc_value = loc[1]
    else:
      loc_value = ""
    abs_value = abs(value)
    deg = int(abs_value)
    t1 = (abs_value - deg) * 60
    min = int(t1)
    sec = round((t1 - min) * 60, 5)
    return (deg, min, sec, loc_value)

# import piexif
# def set_gps_location(file_name, lat, lng, altitude, gpsTime, meta=None):
#     """Adds GPS position as EXIF metadata
#     Keyword arguments:
#     file_name -- image file
#     lat -- latitude (as float)
#     lng -- longitude (as float)
#     altitude -- altitude (as float)
#     """
#     exif_dict = piexif.load(file_name)
#     lat_deg = to_deg(lat, ["S", "N"])
#     lng_deg = to_deg(lng, ["W", "E"])

#     exiv_lat = (change_to_rational(lat_deg[0]), change_to_rational(lat_deg[1]), change_to_rational(lat_deg[2]))
#     exiv_lng = (change_to_rational(lng_deg[0]), change_to_rational(lng_deg[1]), change_to_rational(lng_deg[2]))

#     gps_ifd = {
#       piexif.GPSIFD.GPSVersionID: (2, 0, 0, 0),
#       piexif.GPSIFD.GPSAltitudeRef: 0,
#       piexif.GPSIFD.GPSAltitude: change_to_rational(round(altitude)),
#       piexif.GPSIFD.GPSLatitudeRef: lat_deg[3],
#       piexif.GPSIFD.GPSLatitude: exiv_lat,
#       piexif.GPSIFD.GPSLongitudeRef: lng_deg[3],
#       piexif.GPSIFD.GPSLongitude: exiv_lng,
#       piexif.GPSIFD.GPSDateStamp: gpsTime
#     }
#     if meta:
#         exif_dict = {"GPS": gps_ifd,
#                     "DJI GimbalRollDegree":meta["attitude"]["roll"],
#                     "DJI GimbalYawDegree": meta["attitude"]["yaw"],
#                     "DJI GimbalPitchDegree": meta["attitude"]["pitch"]
#         }
#     else:
#         exif_dict = {"GPS": gps_ifd}

#     exif_bytes = piexif.dump(exif_dict)
#     piexif.insert(exif_bytes, file_name)

# def insert_exif(file_name,exif_dict):
#     exif_bytes = piexif.dump(exif_dict)
#     piexif.insert(exif_bytes, file_name)

def create_dir(dir_):
    os.makedirs(dir_, exist_ok = True)

def split_list(list_a, chunk_size):
    for i in range(0, len(list_a), chunk_size):
        yield list_a[i:i + chunk_size]

def warp_image(rgb_img, ir_img, meta, lens_offset= 0.01, obj_distance=10.0):

    # Rotate image based on image width and height
    # We want to have the image in portrait mode
    image_width = meta["ImageWidth"]
    image_height = meta["ImageHeight"]

    if obj_distance == 0:
        obj_distance = 10.0

    if image_width > image_height:
        ir_img = cv2.rotate(ir_img, cv2.ROTATE_90_CLOCKWISE)
        rgb_img = cv2.rotate(rgb_img, cv2.ROTATE_90_CLOCKWISE)
        # Now the image is 480 width and 640 height
        # Get the offset of the image. OffsetY is width axis, OffsetX is height axis
        offsetX = int(meta["OffsetY"])
        offsetY = int(meta["OffsetX"])
    else:
        offsetX = int(meta["OffsetX"])
        offsetY = int(meta["OffsetY"])
        
    # Get the real2ir value
    real2ir = float(meta["Real2IR"])

    # Get the IR Crop window using Real2IR value, RGB image image width and height and offset values from the center
    # The crop window is in the form of (x1, y1, x2, y2)
    rgb_img_width = rgb_img.shape[1]
    rgb_img_height = rgb_img.shape[0]
    crop_window_w = rgb_img_width/real2ir
    crop_window_h = rgb_img_height/real2ir
    crop_window = (int(rgb_img_width/2 - crop_window_w/2), int(rgb_img_height/2 - crop_window_h/2), int(rgb_img_width/2 + crop_window_w/2), int(rgb_img_height/2 + crop_window_h/2))
    # Apply offset
    if 1:
        # Adjust offset based on the lens distance and object distance
        fx_rgb = fy_rgb = 1180
        offsetY_default = 32
        min_obj_distance = lens_offset / ((rgb_img_height - crop_window[3] - offsetY_default)/fy_rgb)
        obj_distance = max(obj_distance, min_obj_distance)
        #offsetY = offsetY_default + int(lens_offset / obj_distance * fy_rgb)
        offsetY = offsetY_default
        crop_window = (crop_window[0] + offsetX, crop_window[1] + offsetY, crop_window[2] + offsetX, crop_window[3] + offsetY)
    
    # Crop the RGB image using the crop window
    rgb_flir_cropped = rgb_img[crop_window[1]:crop_window[3], crop_window[0]:crop_window[2], :]
    # Resize the RGB image to the size of the IR image
    rgb_flir_resized = cv2.resize(rgb_flir_cropped, (ir_img.shape[1], ir_img.shape[0]))

    # Draw rect for debug
    rgb_img_disp = rgb_img.copy()
    cv2.rectangle(rgb_img_disp, (crop_window[0],crop_window[1]), (crop_window[2],crop_window[3]), (0, 255, 0), thickness=2)

    return rgb_flir_resized, rgb_img_disp


# def warp_image(rgb_img, ir_img, dist=0.35, X_offset_pix = 0.00, Y_offset = 0.01):
#     rgb_shape = rgb_img.shape
#     X = []
#     Y = []
#     Z = dist

#     ir_shape = ir_img.shape



#     dist = max(dist, 0.1)
#     if 1:
#         # Image to world, 4 corners
#         fx_ir = fy_ir = 665
#         if 0:
#             cx_ir = 230
#             cy_ir = 334
#         else:
#             cx_ir = 240
#             cy_ir = 320

#         for ix in [0,ir_shape[1]]:
#             for iy in [0,ir_shape[0]]:
#                 X.append((ix-cx_ir+X_offset_pix)/fx_ir * Z )
#                 Y.append((iy-cy_ir)/fy_ir * Z - Y_offset)
        
#         # World to Image
#         fx_rgb = fy_rgb = 1180
#         if 0:
#             cx_rgb = 555
#             cy_rgb = 691
#         else:
#             cx_rgb = 540
#             cy_rgb = 720

#         res_X = []
#         res_Y = []
#         for ix,iy in zip(X,Y):
#             res_X.append(int((ix*fx_rgb + cx_rgb*Z)/Z))
#             res_Y.append(int((iy*fy_rgb + cy_rgb*Z)/Z))
        
#         # Draw rect for debug
#         rgb_img_disp = rgb_img.copy()
#         cv2.rectangle(rgb_img_disp, (res_X[0],res_Y[0]), (res_X[3],res_Y[3]), (0, 255, 0), thickness=2)


#         res = rgb_img[res_Y[0]:res_Y[1],res_X[0]:res_X[2],:]
#     else:
#         #rgb_img = cv2.resize(rgb_img,dsize=(0,0), fx=1/1.26241874694824,fy=1/1.26241874694824)
#         Offset_X = -8
#         Offset_Y = 32
#         # Draw rect for debug
#         res_X = []
#         res_Y = []
#         res_X.append(int(rgb_img.shape[1]/2-rgb_img.shape[1]/2/1.26241874694824 - Offset_X))
#         res_Y.append(int(rgb_img.shape[0]/2-rgb_img.shape[0]/2/1.26241874694824 - Offset_Y))

#         res_X.append(int(rgb_img.shape[1]/2+rgb_img.shape[1]/2/1.26241874694824 - Offset_X))
#         res_Y.append(int(rgb_img.shape[0]/2+rgb_img.shape[0]/2/1.26241874694824 - Offset_Y))

#         rgb_img_disp = rgb_img.copy()
#         cv2.rectangle(rgb_img_disp, (res_X[0],res_Y[0]), (res_X[1],res_Y[1]), (0, 255, 0), thickness=2)


#         res = rgb_img[res_Y[0] :res_Y[1],res_X[0] :res_X[1],:]
#     # Resize
#     res = cv2.resize(res, dsize=(ir_shape[1],ir_shape[0]))


#     return res, rgb_img_disp

# def process_item(flir_path, save_dir_RGB, save_dir_IR,iDataset=None):
#     fie = FlirImageExtractor(exiftool_path="exiftool", is_debug=False)

#     fie.process_image(flir_path)
#     thermal_flir = fie.get_thermal_np()

#     #rgb_flir = cv2.cvtColor(fie.extract_embedded_image(),cv2.COLOR_RGB2BGR)
#     rgb_flir = fie.extract_embedded_image()

#     offset_init = 0.01

#     # Create the figure and the line that we will manipulate
#     fig, ax = plt.subplots(ncols=4)
#     if 0:
#         ax[0].imshow(thermal_flir,cmap='gray')
#     else:
#         flir_disp = cv2.applyColorMap(disp_depth(thermal_flir), cv2.COLORMAP_JET)
#         ax[0].imshow(flir_disp)

#     ax[0].axis("off")
#     ax[1].imshow(rgb_flir)
#     ax[1].axis("off")
    
   
#     # adjust the main plot to make room for the sliders
#     fig.subplots_adjust(bottom=0.3)

#     # Make a horizontal slider to control y0
#     axfreq = fig.add_axes([0.2, 0.15, 0.6, 0.03]) #arguments are: [left_position, bottom_position, width, height]
#     lens_offset_slider = Slider(
#         ax=axfreq,
#         #label='$lensoffset$',
#         label='lens_offset',      
#         valmin=-0.03,
#         valmax=0.03,
#         valinit=offset_init,
#         #valinit=0.15,
#     )
    
#     axfreq = fig.add_axes([0.2, 0.1, 0.6, 0.03]) #arguments are: [left_position, bottom_position, width, height]
#     obj_dist_slider = Slider(
#         ax=axfreq,
#         #label='$obj dist$',
#         label='obj_dist$',
#         valmin=0.01,
#         valmax=20,
#         valinit=0.35,
#         #valinit=0.01,
#     )

#     # The function to be called anytime a slider's value changes
#     def update(val):
#         #print(val)
#         dist = lens_offset_slider.val
#         #res,rgb_img_disp = warp_image(rgb_flir, thermal_flir, dist, X_offset_pix=offsetX, Y_offset=Yoffset_slider.val)
#         res,rgb_img_disp = warp_image(rgb_flir, thermal_flir, fie.meta, obj_distance=obj_dist_slider.val, lens_offset=dist)
#         ax[1].imshow(rgb_img_disp)
#         ax[1].axis("off")
#         ax[2].imshow(res)
#         ax[2].axis("off")
        

#         # Mix image
#         # flir_disp = cv2.cvtColor(disp_depth(thermal_flir),cv2.COLOR_GRAY2BGR)
#         # Apply the colormap
#         flir_disp = cv2.applyColorMap(disp_depth(thermal_flir), cv2.COLORMAP_JET)

#         # Blend the images using cv2.addWeighted()
#         blended_image = cv2.addWeighted(flir_disp, 0.2, res, 0.8, 0)
#         ax[3].imshow(blended_image)
#         ax[3].axis("off")


#         fig.canvas.draw_idle()
    
#     update(0)

#     lens_offset_slider.on_changed(update)
#     obj_dist_slider.on_changed(update)
#     plt.show()

if __name__ == "__main__":

    root_dir = ""
    #root_dir = "/Volumes/usbshare2/work/GEMINI/Thermal_camera/FLIR_One_calibration/FLIR_One_Calibration/flir_jpg"
    #root_dir = "/Users/lion397/data/Dataset_2023/Davis/2023-05-30/Drone/iPhone/flir_jpg"

    #flir_files = get_file_list(root_dir, subdir="flir_jpg", exts=["jpg"])
    # flir_files = get_file_list(root_dir, subdir="", exts=["jpg"])


    
    # process_item("/data3/Dataset_2023/Davis/2023-06-20/Drone/iPhone/flir_jpg/230620_IMG_00140.jpg",None,None,None)