#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# https://github.com/Nervengift/read_thermal.py

from __future__ import print_function

import argparse
import io
import json
import os
import os.path
import re
import csv
import subprocess
from PIL import Image
from math import sqrt, exp, log
try:
    from matplotlib import cm
    from matplotlib import pyplot as plt
except Exception as e:
    pass

import numpy as np

import time
import pandas as pd

class FlirImageExtractor:

    def __init__(self, exiftool_path="exiftool", use_calibration_data=True, is_debug=False):
        self.exiftool_path = exiftool_path
        self.is_debug = is_debug
        self.flir_img_filename = ""
        self.image_suffix = "_rgb_image.jpg"
        self.thumbnail_suffix = "_rgb_thumb.jpg"
        self.thermal_suffix = "_thermal.png"
        self.default_distance = 1.0

        # valid for PNG thermal images
        self.use_thumbnail = False
        self.fix_endian = True

        self.rgb_image_np = None
        self.thermal_image_np = None
        self.mean_thermal_np = 0
        self.RawThermalImage = None

        self.meta = None
        
        # Try to load the caliration data if it exists
        if use_calibration_data:
            try:
                # Load the calibration data. The calibration data is located in the same folder as this script
                calibration_data_path = os.path.join(os.path.dirname(__file__), 'calibration_data.csv')
                self.calibration_data = pd.read_csv(calibration_data_path)
                # print(self.calibration_data)
            except Exception as e:
                print(e)
                print("No calibration data found")
                self.calibration_data = None
        else:
            self.calibration_data = None
    pass

    def process_image(self, flir_img_filename, new_meta=None):
        """
        Given a valid image path, process the file: extract real thermal values
        and a thumbnail for comparison (generally thumbnail is on the visible spectre)
        :param flir_img_filename:
        :return:
        """
        if self.is_debug:
            print("INFO Flir image filepath:{}".format(flir_img_filename))

        if not os.path.isfile(flir_img_filename):
            raise ValueError("Input file does not exist or this user don't have permission on this file")

        self.flir_img_filename = flir_img_filename

        if self.get_image_type().upper().strip() == "TIFF":
            # valid for tiff images from Zenmuse XTR
            self.use_thumbnail = True
            self.fix_endian = False

        self.meta = self.extract_metadata()
        if new_meta is not None:
            self.update_meta(new_meta)
        self.rgb_image_np = self.extract_embedded_image()
        self.thermal_image_np = self.extract_thermal_image()

    def get_image_type(self):
        """
        Get the embedded thermal image type, generally can be TIFF or PNG
        :return:
        """
        meta_json = subprocess.check_output(
            [self.exiftool_path, '-RawThermalImageType', '-j', self.flir_img_filename])
        meta = json.loads(meta_json.decode())[0]

        return meta['RawThermalImageType']
    
    def get_rgb_np(self):
        """
        Return the last extracted rgb image
        :return:
        """
        return self.rgb_image_np

    def get_thermal_np(self):
        """
        Return the last extracted thermal image
        :return:
        """
        return self.thermal_image_np
    
    def get_RawThermalImage(self):
        """
        Return the last extracted raw thermal image
        :return:
        """
        return self.RawThermalImage

    def extract_metadata(self):
        """
        Extract embedded thermal image metadata
        Read image metadata needed for conversion of the raw sensor values
        :return:
        """
        # E=1,SD=1,RTemp=20,ATemp=RTemp,IRWTemp=RTemp,IRT=1,RH=50,PR1=21106.77,PB=1501,PF=1,PO=-7340,PR2=0.012545258
        meta_json = subprocess.check_output(
            [self.exiftool_path, self.flir_img_filename, '-CameraSerialNumber','-Emissivity', '-SubjectDistance', '-AtmosphericTemperature',
             '-ReflectedApparentTemperature', '-IRWindowTemperature', '-IRWindowTransmission', '-RelativeHumidity',
             '-PlanckR1', '-PlanckB', '-PlanckF', '-PlanckO', '-PlanckR2',
             '-OffsetX','-OffsetY', "-Orientation","-Real2IR", "-ImageWidth", "-ImageHeight",
               '-j'])
        meta = json.loads(meta_json.decode())[0]
        if self.is_debug:
            print(meta)

        # Update meta string to floats
        meta['ReflectedApparentTemperature'] = FlirImageExtractor.extract_float(meta['ReflectedApparentTemperature'])
        meta['AtmosphericTemperature'] = FlirImageExtractor.extract_float(meta['AtmosphericTemperature'])
        meta['IRWindowTemperature'] = FlirImageExtractor.extract_float(meta['IRWindowTemperature'])
        meta['RelativeHumidity'] = FlirImageExtractor.extract_float(meta['RelativeHumidity'])
        if 'SubjectDistance' in meta:
            meta['SubjectDistance'] = FlirImageExtractor.extract_float(meta['SubjectDistance'])
            
               
        # Check the camera serial number is in the self.calibration_data
        if self.calibration_data is not None:
            if meta['CameraSerialNumber'] in self.calibration_data['SN'].values:
                # Get the calibration data
                calibration_data = self.calibration_data[self.calibration_data['SN'] == meta['CameraSerialNumber']]
                # print(calibration_data)
                meta['PlanckR1'] = calibration_data['PR1'].values[0]
                meta['PlanckO'] = calibration_data['PO'].values[0]
                # print(meta)
            else:
                print(f"Serial number {meta['CameraSerialNumber']} not found in the calibration data")

        return meta

    def update_meta(self, new_meta):
        """
        Update the thermal environment parameters
        :param air_temp: Air temperature in Celsius
        :param air_humidity: Air humidity in percentage
        :param distance: Distance in meters
        :param emissivity: Emissivity of the object
        :param transmission: Transmission of the IR window
        :return:
        air_temp: float, air_humidity: float, distance: float, emissivity: float, transmission: float
        """
        for key, value in new_meta.items():
            self.meta[key] = value

    def extract_embedded_image(self):
        """
        extracts the visual image as 2D numpy array of RGB values
        """
        image_tag = "-EmbeddedImage"
        if self.use_thumbnail:
            image_tag = "-ThumbnailImage"

        visual_img_bytes = subprocess.check_output([self.exiftool_path, image_tag, "-b", self.flir_img_filename])
        visual_img_stream = io.BytesIO(visual_img_bytes)

        visual_img = Image.open(visual_img_stream)
        visual_np = np.array(visual_img)

        return visual_np

    def extract_thermal_image(self):
        """
        extracts the thermal image as 2D numpy array with temperatures in oC
        """

        meta = self.meta

        # exifread can't extract the embedded thermal image, use exiftool instead
        thermal_img_bytes = subprocess.check_output([self.exiftool_path, "-RawThermalImage", "-b", self.flir_img_filename])
        thermal_img_stream = io.BytesIO(thermal_img_bytes)

        thermal_img = Image.open(thermal_img_stream)
        thermal_np = np.array(thermal_img)

        # Measure time if debug is enabled
        if self.is_debug:
            print("INFO Extracting thermal image took {}s".format(time.time() - start_time))
            start_time = time.time()

        subject_distance = self.default_distance
        if 'SubjectDistance' in meta:
            subject_distance = meta['SubjectDistance']
        else:
            subject_distance = self.default_distance

        # raw values -> temperature
        if self.fix_endian:
            # fix endianness, the bytes in the embedded png are in the wrong order
            #thermal_np = np.vectorize(lambda x: (x >> 8) + ((x & 0x00ff) << 8))(thermal_np)
            thermal_np = (thermal_np >> 8) + ((thermal_np & 0x00ff) << 8)
            pass
        
        self.RawThermalImage = thermal_np # Back up the raw thermal image
        mean_value = np.mean(thermal_np)
        
        if 1:
            # print(mean_value)
            if np.max(thermal_np) < 7500:
                # print(f"mean_value: {mean_value}, Image needs to be Corrected")
                # Add offset
                thermal_np = thermal_np*3.356422717437227 - 1404.6130301150079
            else:
                # Store the mean value for later use
                self.mean_thermal_np = mean_value

        # Measure time if debug is enabled
        if self.is_debug:
            print("INFO Correcting took {}s".format(time.time() - start_time))
            start_time = time.time()

        # Convert to real temperature   
        thermal_np = self.raw2temp(thermal_np, E=meta['Emissivity'], OD=subject_distance,
                                                                        RTemp= meta['ReflectedApparentTemperature'],
                                                                        ATemp= meta['AtmosphericTemperature'],
                                                                        IRWTemp= meta['IRWindowTemperature'],
                                                                        IRT= meta['IRWindowTransmission'],
                                                                        RH=meta['RelativeHumidity'],
                                                                        PR1=meta['PlanckR1'], PB=meta['PlanckB'],
                                                                        PF=meta['PlanckF'],
                                                                        PO=meta['PlanckO'], PR2=meta['PlanckR2'])

        # Measure time if debug is enabled
        if self.is_debug:
            print("INFO Converting to temperature took {}s".format(time.time() - start_time))
            start_time = time.time()
        return thermal_np

    @staticmethod
    def raw2temp(raw, E=1, OD=1, RTemp=20, ATemp=20, IRWTemp=20, IRT=1, RH=50, PR1=21106.77, PB=1501, PF=1, PO=-7340,
                 PR2=0.012545258):
        """
        convert raw values from the flir sensor to temperatures in C
        # this calculation has been ported to python from
        # https://github.com/gtatters/Thermimage/blob/master/R/raw2temp.R
        # a detailed explanation of what is going on here can be found there
        """

        # constants
        ATA1 = 0.006569
        ATA2 = 0.01262
        ATB1 = -0.002276
        ATB2 = -0.00667
        ATX = 1.9

        # transmission through window (calibrated)
        emiss_wind = 1 - IRT
        refl_wind = 0

        # transmission through the air
        h2o = (RH / 100) * exp(1.5587 + 0.06939 * (ATemp) - 0.00027816 * (ATemp) ** 2 + 0.00000068455 * (ATemp) ** 3)
        tau1 = ATX * exp(-sqrt(OD / 2) * (ATA1 + ATB1 * sqrt(h2o))) + (1 - ATX) * exp(
            -sqrt(OD / 2) * (ATA2 + ATB2 * sqrt(h2o)))
        tau2 = ATX * exp(-sqrt(OD / 2) * (ATA1 + ATB1 * sqrt(h2o))) + (1 - ATX) * exp(
            -sqrt(OD / 2) * (ATA2 + ATB2 * sqrt(h2o)))

        # radiance from the environment
        raw_refl1 = PR1 / (PR2 * (exp(PB / (RTemp + 273.15)) - PF)) - PO
        raw_refl1_attn = (1 - E) / E * raw_refl1
        raw_atm1 = PR1 / (PR2 * (exp(PB / (ATemp + 273.15)) - PF)) - PO
        raw_atm1_attn = (1 - tau1) / E / tau1 * raw_atm1
        raw_wind = PR1 / (PR2 * (exp(PB / (IRWTemp + 273.15)) - PF)) - PO
        raw_wind_attn = emiss_wind / E / tau1 / IRT * raw_wind
        raw_refl2 = PR1 / (PR2 * (exp(PB / (RTemp + 273.15)) - PF)) - PO
        raw_refl2_attn = refl_wind / E / tau1 / IRT * raw_refl2
        raw_atm2 = PR1 / (PR2 * (exp(PB / (ATemp + 273.15)) - PF)) - PO
        raw_atm2_attn = (1 - tau2) / E / tau1 / IRT / tau2 * raw_atm2
        raw_obj = (raw / E / tau1 / IRT / tau2 - raw_atm1_attn -
                   raw_atm2_attn - raw_wind_attn - raw_refl1_attn - raw_refl2_attn)

        # temperature from radiance
        temp_celcius = PB / np.log(PR1 / (PR2 * (raw_obj + PO)) + PF) - 273.15
        return temp_celcius

    @staticmethod
    def extract_float(dirtystr):
        """
        Extract the float value of a string, helpful for parsing the exiftool data
        :return:
        """
        digits = re.findall(r"[-+]?\d*\.\d+|\d+", dirtystr)
        return float(digits[0])
    
    def extract_gps_info(self):
        """
        Extract GPS info from the image
        :return:
        """
        meta_json = subprocess.check_output(
            [self.exiftool_path, self.flir_img_filename, '-GPSLatitude', '-GPSLongitude', '-GPSAltitude', '-j'])
    
        meta = json.loads(meta_json.decode())[0]

        return meta

    def plot(self):
        """
        Plot the rgb + thermal image (easy to see the pixel values)
        :return:
        """
        rgb_np = self.get_rgb_np()
        thermal_np = self.get_thermal_np()

        plt.subplot(1, 2, 1)
        plt.imshow(thermal_np, cmap='hot')
        plt.subplot(1, 2, 2)
        plt.imshow(rgb_np)
        plt.show()

    def save_images(self):
        """
        Save the extracted images
        :return:
        """
        rgb_np = self.get_rgb_np()
        thermal_np = self.extract_thermal_image()

        img_visual = Image.fromarray(rgb_np)
        thermal_normalized = (thermal_np - np.amin(thermal_np)) / (np.amax(thermal_np) - np.amin(thermal_np))
        img_thermal = Image.fromarray(np.uint8(cm.inferno(thermal_normalized) * 255))

        fn_prefix, _ = os.path.splitext(self.flir_img_filename)
        thermal_filename = fn_prefix + self.thermal_suffix
        image_filename = fn_prefix + self.image_suffix
        if self.use_thumbnail:
            image_filename = fn_prefix + self.thumbnail_suffix

        if self.is_debug:
            print("DEBUG Saving RGB image to:{}".format(image_filename))
            print("DEBUG Saving Thermal image to:{}".format(thermal_filename))

        img_visual.save(image_filename)
        img_thermal.save(thermal_filename)

    def export_thermal_to_csv(self, csv_filename):
        """
        Convert thermal data in numpy to json
        :return:
        """

        with open(csv_filename, 'w') as fh:
            writer = csv.writer(fh, delimiter=',')
            writer.writerow(['x', 'y', 'temp (c)'])

            pixel_values = []
            for e in np.ndenumerate(self.thermal_image_np):
                x, y = e[0]
                c = e[1]
                pixel_values.append([x, y, c])

            writer.writerows(pixel_values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract and visualize Flir Image data')
    parser.add_argument('-i', '--input', type=str, help='Input image. Ex. img.jpg', required=True)
    parser.add_argument('-p', '--plot', help='Generate a plot using matplotlib', required=False, action='store_true')
    parser.add_argument('-exif', '--exiftool', type=str, help='Custom path to exiftool', required=False,
                        default='exiftool')
    parser.add_argument('-csv', '--extractcsv', help='Export the thermal data per pixel encoded as csv file',
                        required=False)
    parser.add_argument('-d', '--debug', help='Set the debug flag', required=False,
                        action='store_true')
    args = parser.parse_args()

    fie = FlirImageExtractor(exiftool_path=args.exiftool, is_debug=args.debug)
    fie.process_image(args.input)

    if args.plot:
        fie.plot()

    if args.extractcsv:
        fie.export_thermal_to_csv(args.extractcsv)

    fie.save_images()
