"""
Roboflow Inference Module
Handles all Roboflow API inference operations for plot analysis
"""

import os
import threading
import re
import tempfile
import shutil
import time
import pandas as pd
import geopandas as gpd
from PIL import Image
from flask import jsonify, request, send_file
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from shapely.geometry import Polygon
import json  

# Global variables for inference tracking
inference_status = {
    'running': False,
    'progress': 0,
    'message': '',
    'error': None,
    'results': None,
    'completed': False
}


def crop_image_with_overlap(image_path, crop_size=640, overlap=32):
    """
    Crop a large image into smaller patches with overlap
    
    Args:
        image_path (str): Path to the image file
        crop_size (int): Size of each crop (default 640x640)
        overlap (int): Overlap between crops in pixels (default 32)
    
    Returns:
        list: List of dictionaries containing crop info and file paths
    """
    try:
        image = Image.open(image_path)
        img_width, img_height = image.size
        
        crops = []
        stride = crop_size - overlap
        
        y_positions = list(range(0, img_height - crop_size + 1, stride))
        if y_positions and y_positions[-1] + crop_size < img_height:
            y_positions.append(img_height - crop_size)
        
        x_positions = list(range(0, img_width - crop_size + 1, stride))
        if x_positions and x_positions[-1] + crop_size < img_width:
            x_positions.append(img_width - crop_size)
        
        # Handle cases where image is smaller than crop_size
        if not y_positions:
            y_positions = [0]
        if not x_positions:
            x_positions = [0]
        
        # Create temporary directory for crops
        temp_dir = tempfile.mkdtemp()
        
        crop_id = 0
        for y in y_positions:
            for x in x_positions:
                # Adjust coordinates if they would go beyond image boundaries
                actual_x = min(x, img_width - crop_size) if img_width >= crop_size else 0
                actual_y = min(y, img_height - crop_size) if img_height >= crop_size else 0
                actual_width = min(crop_size, img_width - actual_x)
                actual_height = min(crop_size, img_height - actual_y)
                
                # Crop the image
                crop_box = (actual_x, actual_y, actual_x + actual_width, actual_y + actual_height)
                crop = image.crop(crop_box)
                
                # If crop is smaller than desired size, pad with white background
                if actual_width < crop_size or actual_height < crop_size:
                    padded_crop = Image.new('RGB', (crop_size, crop_size), (255, 255, 255))
                    padded_crop.paste(crop, (0, 0))
                    crop = padded_crop
                
                # Save crop to temporary file
                crop_filename = f"crop_{crop_id}.jpg"
                crop_path = os.path.join(temp_dir, crop_filename)
                crop.save(crop_path, format='JPEG', quality=85)
                
                crops.append({
                    'crop_id': crop_id,
                    'x_offset': actual_x,
                    'y_offset': actual_y,
                    'width': actual_width,
                    'height': actual_height,
                    'crop_path': crop_path,
                    'temp_dir': temp_dir
                })
                
                crop_id += 1
        
        return crops
        
    except Exception as e:
        print(f"Error cropping image {image_path}: {e}")
        return []


def transform_predictions_to_plot_coordinates(predictions, crop_info):
    """
    Transform prediction coordinates from crop level to plot level
    
    Args:
        predictions (list): List of predictions from a crop
        crop_info (dict): Information about the crop (x_offset, y_offset, etc.)
    
    Returns:
        list: Transformed predictions with plot-level coordinates
    """
    transformed = []
    
    for pred in predictions:
        # Transform coordinates back to plot level
        plot_x = pred.get('x', 0) + crop_info['x_offset']
        plot_y = pred.get('y', 0) + crop_info['y_offset']
        
        transformed_pred = {
            'class': pred.get('class', ''),
            'confidence': pred.get('confidence', 0),
            'x': plot_x,
            'y': plot_y,
            'width': pred.get('width', 0),
            'height': pred.get('height', 0),
            'crop_id': crop_info['crop_id'],
            'model_id': pred.get('model_id'),
            'model_version': pred.get('model_version')
        }
        
        transformed.append(transformed_pred)
    
    return transformed


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) for two bounding boxes
    
    Args:
        box1, box2: Dictionaries with keys 'x', 'y', 'width', 'height'
    
    Returns:
        float: IoU value between 0 and 1
    """
    # Convert center coordinates to corner coordinates
    x1_min = box1['x'] - box1['width'] / 2
    y1_min = box1['y'] - box1['height'] / 2
    x1_max = box1['x'] + box1['width'] / 2
    y1_max = box1['y'] + box1['height'] / 2
    
    x2_min = box2['x'] - box2['width'] / 2
    y2_min = box2['y'] - box2['height'] / 2
    x2_max = box2['x'] + box2['width'] / 2
    y2_max = box2['y'] + box2['height'] / 2
    
    # Calculate intersection
    intersection_x_min = max(x1_min, x2_min)
    intersection_y_min = max(y1_min, y2_min)
    intersection_x_max = min(x1_max, x2_max)
    intersection_y_max = min(y1_max, y2_max)
    
    if intersection_x_max <= intersection_x_min or intersection_y_max <= intersection_y_min:
        return 0.0
    
    intersection_area = (intersection_x_max - intersection_x_min) * (intersection_y_max - intersection_y_min)
    
    # Calculate union
    box1_area = box1['width'] * box1['height']
    box2_area = box2['width'] * box2['height']
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area


def apply_nms(predictions, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to remove duplicate detections
    
    Args:
        predictions (list): List of predictions with coordinates and confidence
        iou_threshold (float): IoU threshold for considering boxes as duplicates
    
    Returns:
        list: Filtered predictions after NMS
    """
    if not predictions:
        return []
    
    # Group predictions by class
    class_groups = {}
    for pred in predictions:
        class_name = pred['class']
        if class_name not in class_groups:
            class_groups[class_name] = []
        class_groups[class_name].append(pred)
    
    final_predictions = []
    
    # Apply NMS for each class separately
    for class_name, class_predictions in class_groups.items():
        # Sort by confidence (highest first)
        class_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while class_predictions:
            # Take the prediction with highest confidence
            current = class_predictions.pop(0)
            keep.append(current)
            
            # Remove predictions with high IoU overlap
            remaining = []
            for pred in class_predictions:
                iou = calculate_iou(current, pred)
                if iou < iou_threshold:
                    remaining.append(pred)
            
            class_predictions = remaining
        
        final_predictions.extend(keep)
    
    return final_predictions


def create_traits_geojson(predictions_df, data_root_dir, year, experiment, location, population, date, platform, sensor, agrowstitch_dir, plot_filter_enabled=False, selected_populations=None, selected_columns=None, selected_rows=None):
    """
    Create a GeoJSON file with detection counts per plot for integration with the stats/map tabs
    
    Args:
        predictions_df (pd.DataFrame): DataFrame with all predictions
        data_root_dir (str): Root data directory
        year, experiment, location, population, date, platform, sensor, agrowstitch_dir: Dataset parameters
    
    Returns:
        str: Path to created GeoJSON file
    """
    try:
        # Initialize default values for filtering parameters
        if selected_populations is None:
            selected_populations = []
        if selected_columns is None:
            selected_columns = []
        if selected_rows is None:
            selected_rows = []
        
        print(f"DEBUG: Starting create_traits_geojson with parameters:")
        print(f"  data_root_dir: {data_root_dir}")
        print(f"  year: {year}, experiment: {experiment}, location: {location}")
        print(f"  population: {population}, date: {date}, platform: {platform}, sensor: {sensor}")
        print(f"  agrowstitch_dir: {agrowstitch_dir}")
        print(f"  predictions_df shape: {predictions_df.shape}")
        print(f"  plot_filter_enabled: {plot_filter_enabled}")
        print(f"  selected_populations: {selected_populations}")
        print(f"  selected_columns: {selected_columns}")
        print(f"  selected_rows: {selected_rows}")
        
        # Look for existing plot boundary GeoJSON first
        plot_boundary_geojson = os.path.join(
            data_root_dir, 'Intermediate', year, experiment, location, population, 'Plot-Boundary-WGS84.geojson'
        )
        
        print(f"DEBUG: Looking for plot boundary GeoJSON at: {plot_boundary_geojson}")
        
        # If no plot boundary GeoJSON, try to create from plot_borders.csv
        if not os.path.exists(plot_boundary_geojson):
            print(f"DEBUG: Plot boundary GeoJSON not found, looking for plot_borders.csv")
            plot_borders_csv = os.path.join(
                data_root_dir, "Raw", year, experiment, location, population, "plot_borders.csv"
            )
            
            print(f"DEBUG: Looking for plot borders CSV at: {plot_borders_csv}")
            
            if not os.path.exists(plot_borders_csv):
                print(f"ERROR: No plot boundaries found for traits export!")
                print(f"  Missing: {plot_boundary_geojson}")
                print(f"  Missing: {plot_borders_csv}")
                print(f"  Cannot create traits GeoJSON without plot boundaries")
                return None
            
            # Create simple plot boundaries from CSV (this is a simplified approach)
            borders_df = pd.read_csv(plot_borders_csv)
            print(f"DEBUG: Plot borders CSV columns: {list(borders_df.columns)}")
            print("DEBUG: First few rows of borders CSV:")
            print(borders_df.head())
            
            # Create simple rectangular geometries from start/end coordinates
            geometries = []
            for _, row in borders_df.iterrows():
                # Create a simple rectangular polygon from start/end coordinates
                # This is a simplified approach - in reality you'd want proper plot boundaries
                start_lat, start_lon = row['start_lat'], row['start_lon']
                end_lat, end_lon = row['end_lat'], row['end_lon']
                
                # Create a small buffer around the line to make a polygon
                buffer_size = 0.00001  # Small buffer in degrees
                coords = [
                    (start_lon - buffer_size, start_lat - buffer_size),
                    (start_lon + buffer_size, start_lat + buffer_size),
                    (end_lon + buffer_size, end_lat + buffer_size),
                    (end_lon - buffer_size, end_lat - buffer_size),
                    (start_lon - buffer_size, start_lat - buffer_size)
                ]
                geometries.append(Polygon(coords))
            
            # Create GeoDataFrame with plot info
            plot_gdf = gpd.GeoDataFrame({
                'plot_index': borders_df['plot_index'],
                'Plot': borders_df.get('Plot', borders_df['plot_index']),
                'accession': borders_df.get('Accession', 'Unknown'),  # Map Accession to accession for consistency
                'population': population,
                'geometry': geometries
            }, crs='EPSG:4326')
        else:
            # Load existing plot boundary GeoJSON
            plot_gdf = gpd.read_file(plot_boundary_geojson)
            print(f"DEBUG: Loaded plot boundary GeoJSON with columns: {list(plot_gdf.columns)}")
            print("DEBUG: First few rows of plot GeoJSON:")
            print(plot_gdf[['plot_index'] if 'plot_index' in plot_gdf.columns else plot_gdf.columns[:3]].head())
            
        # Also ensure we have a 'Plot' column for the final output (Stats/Map tabs expect this)
        if 'Plot' not in plot_gdf.columns or plot_gdf['Plot'].isna().all():
            if 'plot' in plot_gdf.columns:
                print("DEBUG: Using 'plot' column to populate 'Plot' column")
                plot_gdf['Plot'] = plot_gdf['plot']
            else:
                plot_gdf['Plot'] = plot_gdf['plot_index']
        
        print(f"DEBUG: Plot GeoDataFrame shape: {plot_gdf.shape}")
        
        # Filter to only include plots with labeled accessions (non-null, non-'Unknown')
        if 'accession' in plot_gdf.columns:
            # Convert to string and handle various empty representations
            plot_gdf['accession'] = plot_gdf['accession'].astype(str)
            labeled_plots = plot_gdf[
                (plot_gdf['accession'].notna()) & 
                (plot_gdf['accession'] != 'Unknown') & 
                (plot_gdf['accession'] != '') &
                (plot_gdf['accession'] != 'NaN') &
                (plot_gdf['accession'] != 'nan') &
                (plot_gdf['accession'].str.strip() != '')  # Handle whitespace-only strings
            ].copy()
            
            print(f"DEBUG: Original plots: {len(plot_gdf)}, Labeled plots: {len(labeled_plots)}")
            print(f"DEBUG: Labeled plot accessions: {labeled_plots['accession'].tolist()}")
            
            if len(labeled_plots) > 0:
                plot_gdf = labeled_plots
                # Create mapping from AgRowStitch plot_index to whether it should be included
                # The predictions use sequential plot_index (0,1,2,3...) from AgRowStitch
                # But plot_gdf has the actual field plot numbers in plot_index
                # We need to map based on position/sequence instead
                # print(f"DEBUG: Plot boundaries plot_index range: {plot_gdf['plot_index'].tolist()}")
            else:
                print("WARNING: No labeled plots found, using all plots")
        else:
            print("WARNING: No accession column found, using all plots")
        
        # Apply custom plot filtering if enabled
        if plot_filter_enabled and (selected_populations or selected_columns or selected_rows):
            print(f"DEBUG: Applying plot filtering - populations: {selected_populations}, columns: {selected_columns}, rows: {selected_rows}")
            original_count = len(plot_gdf)
            
            # Apply population filter
            if selected_populations:
                if 'population' in plot_gdf.columns:
                    plot_gdf = plot_gdf[plot_gdf['population'].isin(selected_populations)]
                    print(f"DEBUG: After population filter: {len(plot_gdf)} plots")
                else:
                    print("WARNING: Population filtering requested but no 'population' column found")
            
            # Apply column filter
            if selected_columns:
                if 'column' in plot_gdf.columns:
                    # Convert to string for comparison in case columns are numeric
                    plot_gdf = plot_gdf[plot_gdf['column'].astype(str).isin([str(c) for c in selected_columns])]
                    print(f"DEBUG: After column filter: {len(plot_gdf)} plots")
                else:
                    print("WARNING: Column filtering requested but no 'column' column found")
            
            # Apply row filter
            if selected_rows:
                if 'row' in plot_gdf.columns:
                    # Convert to string for comparison in case rows are numeric
                    plot_gdf = plot_gdf[plot_gdf['row'].astype(str).isin([str(r) for r in selected_rows])]
                    print(f"DEBUG: After row filter: {len(plot_gdf)} plots")
                else:
                    print("WARNING: Row filtering requested but no 'row' column found")
            
            print(f"DEBUG: Plot filtering complete - reduced from {original_count} to {len(plot_gdf)} plots")
            
            if len(plot_gdf) == 0:
                print("ERROR: Plot filtering resulted in no plots to process")
                inference_status.update({
                    'running': False,
                    'error': 'Plot filtering resulted in no plots to process. Please adjust your filter criteria.',
                    'completed': True
                })
                return
        
        # Aggregate predictions by plot, skipping rows with missing plot_label or accession
        if not predictions_df.empty:
            # Remove rows with missing plot_label or accession
            filtered_preds = predictions_df.dropna(subset=['plot_label', 'accession'])
            if filtered_preds.empty:
                print("WARNING: All predictions missing plot_label or accession. No stats will be generated.")
            else:
                # Normalize plot_label and Plot to string and strip trailing .0 for matching
                def normalize_plot_label(val):
                    if pd.isna(val):
                        return None
                    sval = str(val).strip()
                    if sval.endswith('.0'):
                        sval = sval[:-2]
                    return sval

                filtered_preds['plot_label_norm'] = filtered_preds['plot_label'].apply(normalize_plot_label)
                plot_gdf['Plot_norm'] = plot_gdf['plot'].apply(normalize_plot_label) if 'plot' in plot_gdf.columns else plot_gdf['Plot'].apply(normalize_plot_label)

                # Map AgRowStitch sequential plot_index to field plot number (Plot column)
                plot_mapping = {}
                for _, row in plot_gdf.iterrows():
                    seq_idx = row['Plot_norm']
                    plot_mapping[seq_idx] = {
                        'Plot': row['plot'],
                        'accession': row.get('Accession', row.get('accession', 'Unknown'))
                    }
                print(f"DEBUG: Plot mapping created: {plot_mapping}")

                # Update predictions to use the correct field plot number and accession
                filtered_preds['field_plot_number'] = filtered_preds['plot_label_norm'].map(
                    lambda x: plot_mapping.get(x, {}).get('Plot', None)
                )
                filtered_preds['accession'] = filtered_preds['plot_label_norm'].map(
                    lambda x: plot_mapping.get(x, {}).get('accession', None)
                )
                # Only keep predictions that map to a valid field plot number
                filtered_preds = filtered_preds[filtered_preds['field_plot_number'].notnull()]
                filtered_preds['accession'] = filtered_preds['field_plot_number'].map(
                    lambda x: plot_mapping.get(normalize_plot_label(x), {}).get('accession', '')
                )

                # Use field_plot_number for grouping and output
                filtered_preds['plot_label'] = filtered_preds['field_plot_number']

                # Calculate total detections per field plot
                total_detections = filtered_preds.groupby('plot_label').size()

                # Get model_id from predictions (should be consistent across all predictions)
                model_id = filtered_preds['model_id'].iloc[0] if 'model_id' in filtered_preds.columns and not filtered_preds['model_id'].isna().all() else 'unknown'

                # Create summary columns for each detected class using model_id/class format
                class_columns = {}
                for class_name in filtered_preds['class'].unique():
                    if pd.notna(class_name):
                        class_counts = filtered_preds[filtered_preds['class'] == class_name].groupby('plot_label').size()
                        # Use model_id/class_label format for trait names
                        trait_name = f"{model_id}/{class_name}" if model_id != 'unknown' else class_name
                        class_columns[trait_name] = class_counts

                print(f"DEBUG: Created class columns: {list(class_columns.keys())}")
                print(f"DEBUG: Total detections by plot: {dict(total_detections)}")

                # Merge with plot boundaries
                traits_gdf = plot_gdf.copy()

                # Add detection counts to the GeoDataFrame (only individual class counts, no total column)
                for idx, row in traits_gdf.iterrows():
                    field_plot_number = row['plot']
                    for trait_name in class_columns.keys():
                        # Always add trait columns, even if zero
                        traits_gdf.at[idx, trait_name] = class_columns[trait_name].get(field_plot_number, 0)

                # Add plot_label and accession columns to output
                traits_gdf['plot_label'] = traits_gdf['plot']
                if 'Accession' in traits_gdf.columns:
                    traits_gdf['accession'] = traits_gdf['Accession']
                elif 'accession' in traits_gdf.columns:
                    traits_gdf['accession'] = traits_gdf['accession']
        
        # Convert detection count columns to int
        detection_columns = [col for col in traits_gdf.columns if 'Total_Detections' in col or '/' in col]
        for col in detection_columns:
            traits_gdf[col] = traits_gdf[col].astype(int)
        
        # Save the traits GeoJSON file at the ORTHOMOSAIC VERSION level (not sensor level)
        # This allows multiple orthomosaic versions to have separate trait files
        if agrowstitch_dir == 'Plot_Images':
            # For Plot_Images, save to plot_images directory
            output_dir = os.path.join(
                data_root_dir, 'Intermediate', year, experiment, location, population, 'plot_images', date
            )
            geojson_filename = f"{date}-Plot_Images-Traits-WGS84.geojson"
        else:
            # For AgRowStitch, use traditional path
            output_dir = os.path.join(
                data_root_dir, 'Processed', year, experiment, location, population,
                date, platform, sensor, agrowstitch_dir
            )
            geojson_filename = f"{date}-{platform}-{sensor}-{agrowstitch_dir}-Traits-WGS84.geojson"
        
        print(f"DEBUG: Creating output directory at orthomosaic level: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Verify directory was created
        if os.path.exists(output_dir):
            print(f"Output directory exists: {output_dir}")
        else:
            print(f"ERROR: Failed to create output directory: {output_dir}")
        
        geojson_path = os.path.join(output_dir, geojson_filename)
        geojson_path = os.path.join(output_dir, geojson_filename)
        
        print(f"DEBUG: Full GeoJSON path will be: {geojson_path}")
        
        # Check if traits file already exists and merge if so
        if os.path.exists(geojson_path):
            try:
                existing_gdf = gpd.read_file(geojson_path)
                
                # Merge on common columns, keeping all existing traits and adding new detection data
                merge_columns = ['plot', 'accession', 'population', 'geometry']
                
                # Only merge on columns that exist in both dataframes
                available_merge_cols = [col for col in merge_columns if col in existing_gdf.columns and col in traits_gdf.columns]
                
                if available_merge_cols:
                    # Update existing records with new detection data
                    for idx, row in traits_gdf.iterrows():
                        # Find matching row in existing data
                        plot_val = row.get('plot')
                        mask = existing_gdf['plot'] == plot_val

                        if mask.any():
                            # Update existing row with new detection data
                            for col in traits_gdf.columns:
                                # Add new detection columns (model_id/class or Total_Detections_*)
                                if (('/' in col and col not in existing_gdf.columns) or 
                                    ('Total_Detections_' in col and col not in existing_gdf.columns)):
                                    existing_gdf.loc[mask, col] = row[col]
                                # Update existing detection columns
                                elif ('/' in col or 'Total_Detections_' in col):
                                    existing_gdf.loc[mask, col] = row[col]
                    
                    traits_gdf = existing_gdf
                
            except Exception as e:
                print(f"Warning: Could not merge with existing traits file: {e}")
        
        # Remove 'Plot' and 'plot_label' columns and reorder for stats table
        stats_columns = ['column', 'row', 'location', 'population', 'accession', 'plot']
        # Remove if present
        for col in ['Plot', 'plot_label', 'Plot_norm']:
            if col in traits_gdf.columns:
                traits_gdf = traits_gdf.drop(columns=[col])
        # Reorder columns: stats columns, then detection columns, then the rest
        available_stats_cols = [col for col in stats_columns if col in traits_gdf.columns]
        detection_cols = [col for col in traits_gdf.columns if ('/' in col or 'Total_Detections' in col) and col not in available_stats_cols]
        other_cols = [col for col in traits_gdf.columns if col not in available_stats_cols + detection_cols]
        traits_gdf = traits_gdf[available_stats_cols + detection_cols + other_cols]

        # Sort by 'column' if it exists
        if 'column' in traits_gdf.columns:
            traits_gdf = traits_gdf.sort_values(by='column').reset_index(drop=True)

        # Save the final GeoJSON
        traits_gdf.to_file(geojson_path, driver='GeoJSON')
        print(f"Saved traits GeoJSON: {geojson_path}")
        
        # Verify the file was actually created
        if os.path.exists(geojson_path):
            print(f"GeoJSON file successfully created and verified at: {geojson_path}")
            print(f"File size: {os.path.getsize(geojson_path)} bytes")
        else:
            print(f"ERROR: GeoJSON file was not created at: {geojson_path}")
        
        return geojson_path
        
    except Exception as e:
        print(f"Error creating traits GeoJSON: {e}")
        return None


def run_roboflow_inference_endpoint(file_app, data_root_dir):
    """
    Run Roboflow inference on stitched plot images
    """
    @file_app.route('/run_roboflow_inference', methods=['POST'])
    def run_roboflow_inference():
        global inference_status
        try:
            data = request.json
            print(f"[DEBUG] /run_roboflow_inference called. Payload keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
            inference_mode = data.get('inferenceMode', 'cloud')
            model_task = data.get('modelTask', 'detection')  # 'detection' or 'segmentation'
            include_masks = bool(data.get('includeMasks', True))
            print(f"[DEBUG] inference_mode={inference_mode} model_task={model_task} include_masks={include_masks}")
            # ...existing code determining api_url...
            import subprocess
            import socket
            def is_local_inference_running(host='localhost', port=9001):
                try:
                    with socket.create_connection((host, port), timeout=2):
                        return True
                except Exception as e:
                    return False
            if inference_mode == 'local':
                api_url = data.get('apiUrl', 'http://localhost:9001')
                print(f"[DEBUG] Using local inference. api_url={api_url}")
                already_running = is_local_inference_running()
                print(f"[DEBUG] Local inference running before start attempt: {already_running}")
                if not already_running:
                    try:
                        print("[DEBUG] Attempting to start local inference server via 'inference server start'")
                        subprocess.Popen(['inference', 'server', 'start'])
                        max_wait = 600
                        waited = 0
                        while not is_local_inference_running() and waited < max_wait:
                            if waited % 10 == 0:
                                print(f"[DEBUG] Waiting for local inference server... {waited}s")
                            time.sleep(1)
                            waited += 1
                        if not is_local_inference_running():
                            print("[ERROR] Local inference server failed to start within timeout")
                            return jsonify({'error': 'Local inference server failed to start within 10 minutes.'}), 500
                        print(f"[DEBUG] Local inference server started after {waited}s")
                    except FileNotFoundError as fe:
                        print(f"[ERROR] 'inference' CLI not found. Ensure it is installed and on PATH. {fe}")
                        return jsonify({'error': "'inference' CLI not found. Install the Roboflow inference client."}), 500
                    except Exception as e:
                        print(f"[ERROR] Exception starting local inference server: {type(e).__name__}: {e}")
                        return jsonify({'error': f'Failed to start local inference server: {e}'}), 500
            else:
                api_url = data.get('apiUrl', 'https://detect.roboflow.com')
                print(f"[DEBUG] Using cloud inference. api_url={api_url}")
            api_key = data.get('apiKey')
            model_id = data.get('modelId')
            year = data.get('year')
            experiment = data.get('experiment')
            location = data.get('location')
            population = data.get('population')
            date = data.get('date')
            platform = data.get('platform')
            sensor = data.get('sensor')
            agrowstitch_dir = data.get('agrowstitchDir')
            
            # Extract plot filtering parameters
            plot_filter_enabled = data.get('plotFilterEnabled', False)
            selected_populations = data.get('selectedPopulations', [])
            selected_columns = data.get('selectedColumns', [])
            selected_rows = data.get('selectedRows', [])
            
            # Extract confidence threshold parameter (optional)
            confidence_threshold = data.get('confidence_threshold', None)  # Expects value between 0.01 and 0.99, or None
            
            print(f"[DEBUG] Params year={year} experiment={experiment} location={location} population={population} date={date} platform={platform} sensor={sensor} agrowstitchDir={agrowstitch_dir}")
            print(f"[DEBUG] Plot filtering: enabled={plot_filter_enabled}, populations={selected_populations}, columns={selected_columns}, rows={selected_rows}")
            print(f"[DEBUG] Confidence threshold: {confidence_threshold}")
            
            if not all([api_key, model_id, year, experiment, location, population, date, platform, sensor, agrowstitch_dir]):
                print("[ERROR] Missing required parameters for inference start")
                return jsonify({'error': 'Missing required parameters'}), 400
            inference_status.update({'running': True,'progress': 0,'message': 'Initializing inference...','error': None,'results': None,'completed': False})
            print("[DEBUG] Spawning inference worker thread")
            thread = threading.Thread(target=run_roboflow_inference_worker, args=(api_url, api_key, model_id, year, experiment, location, population, date, platform, sensor, agrowstitch_dir, data_root_dir, model_task, include_masks, plot_filter_enabled, selected_populations, selected_columns, selected_rows, confidence_threshold), daemon=True)
            thread.start()
            return jsonify({'message': 'Inference started successfully'}), 200
        except Exception as e:
            print(f"[ERROR] Error starting Roboflow inference: {type(e).__name__}: {e}")
            inference_status['error'] = str(e)
            return jsonify({'error': str(e)}), 500


def run_roboflow_inference_worker(api_url, api_key, model_id, year, experiment, location, population, date, platform, sensor, agrowstitch_dir, data_root_dir, model_task='detection', include_masks=True, plot_filter_enabled=False, selected_populations=None, selected_columns=None, selected_rows=None, confidence_threshold=None):
    """Worker function to run Roboflow inference in background (supports detection & segmentation)."""
    global inference_status
    
    if selected_populations is None:
        selected_populations = []
    if selected_columns is None:
        selected_columns = []
    if selected_rows is None:
        selected_rows = []
    try:
        inference_status['message'] = 'Loading plot images...'
        inference_status['progress'] = 10
        
        # Check if using Plot_Images from split_orthomosaics
        if agrowstitch_dir == 'Plot_Images':
            # Use plot images from split_orthomosaics
            plots_dir = os.path.join(
                data_root_dir, 'Intermediate', year, experiment, location, population, 'plot_images', date
            )
            
            if not os.path.exists(plots_dir):
                inference_status.update({'error': f'Plot images directory not found: {plots_dir}. Please run "Get Plot Images" first.','running': False})
                return
                
            # Find all plot PNG files from split_orthomosaics
            plot_files = [f for f in os.listdir(plots_dir) 
                         if f.startswith('plot_') and f.endswith('.png')]
            
            if not plot_files:
                inference_status.update({'error': 'No plot images found. Please run "Get Plot Images" first.','running': False})
                return
                
            use_plot_images_format = True
            
        else:
            # Build path to AgRowStitch plot images
            plots_dir = os.path.join(
                data_root_dir, 'Processed', year, experiment, location, population,
                date, platform, sensor, agrowstitch_dir
            )
            
            if not os.path.exists(plots_dir):
                inference_status.update({'error': f'AgRowStitch directory not found: {plots_dir}','running': False})
                return
                
            # Find all plot image files
            plot_files = [f for f in os.listdir(plots_dir) 
                         if f.startswith('full_res_mosaic_temp_plot_') and f.endswith('.png')]
            
            if not plot_files:
                inference_status.update({'error': 'No plot images found in AgRowStitch directory','running': False})
                return
                
            use_plot_images_format = False
            
        inference_status.update({'message': f'Found {len(plot_files)} plot images','progress': 20})
        
        # Load plot borders data for plot labels
        plot_borders_path = os.path.join(
            data_root_dir, "Raw", year, experiment, location, population, "plot_borders.csv"
        )
        
        plot_data = {}
        if os.path.exists(plot_borders_path):
            try:
                borders_df = pd.read_csv(plot_borders_path)
                for _, row in borders_df.iterrows():
                    plot_idx = row.get('plot_index')
                    if pd.notna(plot_idx) and plot_idx > 0:
                        plot_data[int(plot_idx)] = {
                            'plot': row.get('Plot') if pd.notna(row.get('Plot')) else None,
                            'accession': row.get('Accession') if pd.notna(row.get('Accession')) else None
                        }
            except Exception as e:
                print(f"Warning: Could not load plot borders data: {e}")
        
        # Initialize results storage
        all_predictions = []
        label_counts = {}
        mask_count_total = 0
        has_any_segmentation = False
        
        # Initialize Roboflow client
        inference_status.update({'message': 'Initializing Roboflow client...','progress': 25})
        try:
            client = InferenceHTTPClient(
                api_url=api_url,
                api_key=api_key
            )
        except Exception as e:
            inference_status.update({'error': f'Failed to initialize Roboflow client: {e}','running': False})
            return
        
        inference_status.update({'message': 'Running inference on plots...','progress': 30})
        for i, plot_file in enumerate(sorted(plot_files)):
            try:
                # Extract plot index from filename
                if use_plot_images_format:
                    # Format: plot_{plot_id}_accession_{accession_id}.png
                    plot_match = re.search(r'plot_(\d+)_accession_', plot_file)
                    if not plot_match:
                        print(f"[WARN] Could not parse plot index from Plot_Images filename {plot_file}")
                        continue
                else:
                    # Format: full_res_mosaic_temp_plot_{plot_id}.png
                    plot_match = re.search(r'temp_plot_(\d+)', plot_file)
                    if not plot_match:
                        print(f"[WARN] Could not parse plot index from AgRowStitch filename {plot_file}")
                        continue
                plot_index = int(plot_match.group(1))
                plot_info = plot_data.get(plot_index, {})
                image_path = os.path.join(plots_dir, plot_file)
                print(f"[DEBUG] Processing plot_index={plot_index} file={plot_file} ({i+1}/{len(plot_files)})")
                inference_status['message'] = f'Processing plot {plot_index} ({i + 1}/{len(plot_files)}) - cropping image...'
                crops = crop_image_with_overlap(image_path, crop_size=640, overlap=32)
                if not crops:
                    print(f"[WARN] No crops generated for {plot_file}")
                    continue
                plot_predictions = []
                temp_dirs = set()
                for j, crop_info in enumerate(crops):
                    try:
                        inference_status['message'] = f'Plot {plot_index} crop {j + 1}/{len(crops)}'
                        temp_dirs.add(crop_info['temp_dir'])
                        print(f"[DEBUG] Inference on crop {j+1}/{len(crops)} path={crop_info['crop_path']} offsets=({crop_info['x_offset']},{crop_info['y_offset']}) size=({crop_info['width']}x{crop_info['height']})")
                        
                        try:
                            # Use custom configuration only if confidence threshold is specified
                            if confidence_threshold is not None:
                                custom_configuration = InferenceConfiguration(confidence_threshold=confidence_threshold)
                                with client.use_configuration(custom_configuration):
                                    result = client.infer(crop_info['crop_path'], model_id=model_id)
                                print(f"[DEBUG] Using custom confidence threshold: {confidence_threshold}")
                            else:
                                # Use Roboflow's optimal default confidence threshold
                                result = client.infer(crop_info['crop_path'], model_id=model_id)
                                print(f"[DEBUG] Using Roboflow's default confidence threshold")
                        except Exception as infer_e:
                            print(f"[ERROR] client.infer failed plot={plot_index} crop={j+1}: {type(infer_e).__name__}: {infer_e}")
                            if 'docker' in str(infer_e).lower():
                                inference_status.update({'error': f'Docker related inference error: {infer_e}','running': False})
                                return
                            continue
                        crop_predictions = result.get('predictions', []) if isinstance(result, dict) else []
                        model_version = None
                        if isinstance(result, dict):
                            model_info = result.get('model')
                            if isinstance(model_info, dict):
                                model_version = model_info.get('version')
                        for pred in crop_predictions:
                            pred['model_id'] = model_id
                            pred['model_version'] = model_version
                        transformed = transform_predictions_to_plot_coordinates(crop_predictions, crop_info)
                        # Fix mask placement: transform segmentation points/segments to plot coordinates
                        for tp in transformed:
                            matching = next((p for p in crop_predictions if abs((p.get('x',0)+crop_info['x_offset'])-tp['x'])<1e-3 and abs((p.get('y',0)+crop_info['y_offset'])-tp['y'])<1e-3), None)
                            if matching and model_task == 'segmentation' and include_masks:
                                # Transform points if present
                                if 'points' in matching and matching['points']:
                                    # Each point is {x, y} in crop coordinates
                                    tp['points'] = [
                                        {'x': pt['x'] + crop_info['x_offset'], 'y': pt['y'] + crop_info['y_offset']} for pt in matching['points']
                                    ]
                                elif 'segments' in matching and matching['segments']:
                                    # Each segment is a list of [x, y] pairs
                                    tp['segments'] = [
                                        [pt[0] + crop_info['x_offset'], pt[1] + crop_info['y_offset']] for pt in matching['segments']
                                    ]
                        plot_predictions.extend(transformed)
                    except Exception as e:
                        print(f"[ERROR] Unexpected error processing crop {j} for plot {plot_file}: {type(e).__name__}: {e}")
                        continue
                
                # Clean up temporary crop files
                for temp_dir in temp_dirs:
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception:
                        pass
                
                # Apply Non-Maximum Suppression to remove duplicate detections
                inference_status['message'] = f'Applying NMS to plot {plot_index} predictions...'
                final_predictions = apply_nms(plot_predictions, iou_threshold=0.5)
                
                # Process final predictions for this plot
                for pred in final_predictions:
                    prediction_data = {
                        'plot_index': plot_index,
                        'plot_label': plot_info.get('plot'),
                        'accession': plot_info.get('accession'),
                        'image_file': plot_file,
                        'class': pred.get('class', ''),
                        'confidence': pred.get('confidence', 0),
                        'x': pred.get('x', 0),
                        'y': pred.get('y', 0),
                        'width': pred.get('width', 0),
                        'height': pred.get('height', 0),
                        'model_id': pred.get('model_id'),
                        'model_version': pred.get('model_version')
                    }
                    if model_task == 'segmentation' and include_masks:
                        if 'points' in pred:
                            prediction_data['points'] = json.dumps(pred['points'])
                        elif 'segments' in pred:
                            prediction_data['segments'] = json.dumps(pred['segments'])
                    all_predictions.append(prediction_data)
                    
                    # Count labels
                    cn = pred.get('class','unknown')
                    label_counts[cn] = label_counts.get(cn,0)+1
                
                print(f"Processed plot {plot_index}: {len(crops)} crops -> {len(plot_predictions)} raw predictions -> {len(final_predictions)} final predictions")
                
                # Update progress
                progress = 30 + int((i + 1) / len(plot_files) * 60)
                inference_status.update({'progress': progress,'message': f'Completed plot {plot_index} ({i + 1}/{len(plot_files)})'})
                
            except Exception as e:
                print(f"Error processing {plot_file}: {e}")
                continue
        
        # Save results to CSV
        inference_status['message'] = 'Saving results...'
        inference_status['progress'] = 95
        
        if all_predictions:
            predictions_df = pd.DataFrame(all_predictions)
            
            # Create output directory 
            if agrowstitch_dir == 'Plot_Images':
                # For Plot_Images, save to plot_images directory
                output_dir = os.path.join(
                    data_root_dir, 'Intermediate', year, experiment, location, population, 'plot_images', date
                )
                task_suffix = f"_{model_task}" if model_task else ""
                csv_filename = f"{date}_Plot_Images_roboflow_predictions{task_suffix}.csv"
            else:
                output_dir = os.path.join(
                    data_root_dir, 'Processed', year, experiment, location, population,
                    date, platform, sensor
                )
                task_suffix = f"_{model_task}" if model_task else ""
                csv_filename = f"{date}_{platform}_{sensor}_{agrowstitch_dir}_roboflow_predictions{task_suffix}.csv"
            
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, csv_filename)
            predictions_df.to_csv(csv_path, index=False)
            
            # Create traits GeoJSON for integration with stats/map tabs
            inference_status['message'] = 'Creating traits GeoJSON...'
            geojson_path = create_traits_geojson(
                predictions_df, data_root_dir, year, experiment, location, 
                population, date, platform, sensor, agrowstitch_dir,
                plot_filter_enabled, selected_populations, selected_columns, selected_rows
            )
            
            # Prepare results summary
            label_summary = [{'name': n, 'count': c} for n, c in label_counts.items()]
            
            inference_status['results'] = {
                'csvPath': csv_path,
                'geojsonPath': geojson_path,
                'totalPlots': len(plot_files),
                'totalPredictions': len(all_predictions),
                'labels': label_summary,
                'hasSegmentation': has_any_segmentation,
                'maskCount': mask_count_total
            }
        else:
            inference_status['results'] = {
                'csvPath': None,
                'geojsonPath': None,
                'totalPlots': len(plot_files),
                'totalPredictions': 0,
                'labels': [],
                'hasSegmentation': False,
                'maskCount': 0
            }
        
        inference_status.update({'message': 'Inference completed successfully!','progress': 100,'completed': True,'running': False})
    except Exception as e:
        print(f"Error in Roboflow inference worker: {e}")
        inference_status.update({'error': str(e),'running': False})


def get_inference_progress_endpoint(file_app):
    """
    Get current inference progress
    """
    @file_app.route('/get_inference_progress', methods=['GET'])
    def get_inference_progress():
        return jsonify(inference_status)


def get_agrowstitch_versions_endpoint(file_app, data_root_dir):
    """
    Get available AgRowStitch versions for a specific dataset
    """
    @file_app.route('/get_agrowstitch_versions', methods=['POST'])
    def get_agrowstitch_versions():
        try:
            data = request.json
            year = data.get('year')
            experiment = data.get('experiment')
            location = data.get('location')
            population = data.get('population')
            date = data.get('date')
            platform = data.get('platform')
            sensor = data.get('sensor')
            
            if not all([year, experiment, location, population, date, platform, sensor]):
                return jsonify({'error': 'Missing required parameters'}), 400
                
            # Build path to processed data
            processed_dir = os.path.join(
                data_root_dir, 'Processed', year, experiment, location, population,
                date, platform, sensor
            )
            
            versions = []
            if os.path.exists(processed_dir):
                # Look for AgRowStitch directories (they typically start with 'AgRowStitch' or contain plot images)
                for item in os.listdir(processed_dir):
                    item_path = os.path.join(processed_dir, item)
                    if os.path.isdir(item_path):
                        # Check if directory contains plot images
                        plot_files = [f for f in os.listdir(item_path) 
                                     if f.startswith('full_res_mosaic_temp_plot_') and f.endswith('.png')]
                        if plot_files:
                            versions.append(item)
            
            return jsonify({'versions': sorted(versions)}), 200
            
        except Exception as e:
            print(f"Error getting AgRowStitch versions: {e}")
            return jsonify({'error': str(e)}), 500


def get_inference_results_endpoint(file_app, data_root_dir):
    """
    Get available inference results for browsing and visualization
    """
    @file_app.route('/get_inference_results', methods=['POST'])
    def get_inference_results():
        try:
            data = request.json
            year = data.get('year')
            experiment = data.get('experiment')
            location = data.get('location')
            population = data.get('population')
            
            if not all([year, experiment, location, population]):
                return jsonify({'error': 'Missing required parameters'}), 400
                
            results = []
            
            # Check processed directory for AgRowStitch inference results
            dates_dir = os.path.join(data_root_dir, 'Processed', year, experiment, location, population)
            
            if os.path.exists(dates_dir):
                for date in os.listdir(dates_dir):
                    date_path = os.path.join(dates_dir, date)
                    if not os.path.isdir(date_path):
                        continue
                        
                    for platform in os.listdir(date_path):
                        platform_path = os.path.join(date_path, platform)
                        if not os.path.isdir(platform_path):
                            continue
                            
                        for sensor in os.listdir(platform_path):
                            sensor_path = os.path.join(platform_path, sensor)
                            if not os.path.isdir(sensor_path):
                                continue
                            
                            # Look for inference CSV files (detection and segmentation)
                            for file in os.listdir(sensor_path):
                                if re.match(r'.*_roboflow_predictions(_detection|_segmentation)?\.csv$', file):
                                    csv_path = os.path.join(sensor_path, file)
                                    # Extract agrowstitch version from filename
                                    # Format: date_platform_sensor_agrowstitchversion_roboflow_predictions(_segmentation|_detection).csv
                                    parts = file.split('_')
                                    # Find the index of 'roboflow' to get everything between sensor and 'roboflow'
                                    try:
                                        roboflow_idx = parts.index('roboflow')
                                        orthomosaic_version = '_'.join(parts[3:roboflow_idx]) if roboflow_idx > 3 else 'unknown'
                                    except Exception:
                                        orthomosaic_version = 'unknown'
                                    # Check if corresponding plot images exist
                                    agrowstitch_dir = os.path.join(sensor_path, orthomosaic_version)
                                    plot_images_exist = False
                                    plot_count = 0
                                    if os.path.exists(agrowstitch_dir):
                                        plot_files = [f for f in os.listdir(agrowstitch_dir) 
                                                    if f.startswith('full_res_mosaic_temp_plot_') and f.endswith('.png')]
                                        plot_images_exist = len(plot_files) > 0
                                        plot_count = len(plot_files)
                                    
                                    # Get basic statistics from CSV including model information
                                    total_predictions = 0
                                    classes_detected = []
                                    model_id_used = None
                                    model_version_used = None
                                    has_segmentation = False
                                    mask_count = 0
                                    try:
                                        df = pd.read_csv(csv_path)
                                        total_predictions = len(df)
                                        classes_detected = df['class'].value_counts().to_dict()
                                        # Extract model information from first prediction if available
                                        if not df.empty:
                                            if 'model_id' in df.columns:
                                                model_id_used = df['model_id'].iloc[0] if pd.notna(df['model_id'].iloc[0]) else None
                                            if 'model_version' in df.columns:
                                                model_version_used = df['model_version'].iloc[0] if pd.notna(df['model_version'].iloc[0]) else None
                                            if 'points' in df.columns or 'segments' in df.columns:
                                                # segmentation presence
                                                seg_col = 'points' if 'points' in df.columns else 'segments'
                                                non_null = df[seg_col].dropna()
                                                has_segmentation = not non_null.empty
                                                if has_segmentation:
                                                    # count masks (rows with non-empty json array)
                                                    def count_masks(val):
                                                        try:
                                                            if pd.isna(val): return 0
                                                            parsed = json.loads(val)
                                                            if isinstance(parsed, list):
                                                                return 1 if len(parsed) > 0 else 0
                                                            return 0
                                                        except Exception:
                                                            return 0
                                                    mask_count = int(non_null.apply(count_masks).sum())
                                    except Exception as e:
                                        print(f"Error reading CSV {csv_path}: {e}")
                                    
                                    result_entry = {
                                        'id': f"{date}-{platform}-{sensor}-{orthomosaic_version}",
                                        'date': date,
                                        'platform': platform,
                                        'sensor': sensor,
                                        'orthomosaic': orthomosaic_version,
                                        'model_id': model_id_used,
                                        'model_version': model_version_used,
                                        'csv_path': csv_path,
                                        'total_predictions': total_predictions,
                                        'classes_detected': classes_detected,
                                        'plot_images_available': plot_images_exist,
                                        'plot_count': plot_count,
                                        'timestamp': os.path.getmtime(csv_path),
                                        'has_segmentation': has_segmentation,
                                        'mask_count': mask_count
                                    }
                                    results.append(result_entry)
            
            # Check intermediate directory for Plot_Images inference results
            intermediate_path = os.path.join(data_root_dir, 'Intermediate', year, experiment, location, population, 'plot_images')
            
            if os.path.exists(intermediate_path):
                for date in os.listdir(intermediate_path):
                    date_path = os.path.join(intermediate_path, date)
                    if not os.path.isdir(date_path):
                        continue
                    
                    # Look for inference CSV files in plot_images directory
                    for file in os.listdir(date_path):
                        if re.match(r'.*_Plot_Images_roboflow_predictions(_detection|_segmentation)?\.csv$', file):
                            csv_path = os.path.join(date_path, file)
                            
                            # Check if corresponding plot images exist
                            plot_files = [f for f in os.listdir(date_path) 
                                        if f.startswith('plot_') and f.endswith('.png')]
                            plot_images_exist = len(plot_files) > 0
                            plot_count = len(plot_files)
                            
                            # Get basic statistics from CSV including model information
                            total_predictions = 0
                            classes_detected = []
                            model_id_used = None
                            model_version_used = None
                            has_segmentation = False
                            mask_count = 0
                            try:
                                df = pd.read_csv(csv_path)
                                total_predictions = len(df)
                                classes_detected = df['class'].value_counts().to_dict()
                                # Extract model information from first prediction if available
                                if not df.empty:
                                    if 'model_id' in df.columns:
                                        model_id_used = df['model_id'].iloc[0] if pd.notna(df['model_id'].iloc[0]) else None
                                    if 'model_version' in df.columns:
                                        model_version_used = df['model_version'].iloc[0] if pd.notna(df['model_version'].iloc[0]) else None
                                    if 'points' in df.columns or 'segments' in df.columns:
                                        # segmentation presence
                                        seg_col = 'points' if 'points' in df.columns else 'segments'
                                        non_null = df[seg_col].dropna()
                                        has_segmentation = not non_null.empty
                                        if has_segmentation:
                                            # count masks (rows with non-empty json array)
                                            def count_masks(val):
                                                try:
                                                    if pd.isna(val): return 0
                                                    parsed = json.loads(val)
                                                    if isinstance(parsed, list):
                                                        return 1 if len(parsed) > 0 else 0
                                                    return 0
                                                except Exception:
                                                    return 0
                                            mask_count = int(non_null.apply(count_masks).sum())
                            except Exception as e:
                                print(f"Error reading CSV {csv_path}: {e}")
                            
                            result_entry = {
                                'id': f"{date}-Plot_Images",
                                'date': date,
                                'platform': 'ODM',  # Indicate this is from ODM orthomosaic
                                'sensor': 'RGB',    # Assume RGB for Plot_Images
                                'orthomosaic': 'Plot_Images',
                                'model_id': model_id_used,
                                'model_version': model_version_used,
                                'csv_path': csv_path,
                                'total_predictions': total_predictions,
                                'classes_detected': classes_detected,
                                'plot_images_available': plot_images_exist,
                                'plot_count': plot_count,
                                'timestamp': os.path.getmtime(csv_path),
                                'has_segmentation': has_segmentation,
                                'mask_count': mask_count
                            }
                            results.append(result_entry)
            
            # Sort results by timestamp (newest first)
            results.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return jsonify({'results': results}), 200
            
        except Exception as e:
            print(f"Error getting inference results: {e}")
            return jsonify({'error': str(e)}), 500


def get_plot_predictions_endpoint(file_app, data_root_dir):
    """
    Get predictions for a specific plot image to overlay bounding boxes
    """
    @file_app.route('/get_plot_predictions', methods=['POST'])
    def get_plot_predictions():
        try:
            data = request.json
            year = data.get('year'); experiment = data.get('experiment'); location = data.get('location'); population = data.get('population'); date = data.get('date'); platform = data.get('platform'); sensor = data.get('sensor')
            agrowstitch_version = data.get('agrowstitch_version'); orthomosaic = data.get('orthomosaic')
            version_identifier = orthomosaic if orthomosaic else agrowstitch_version
            plot_filename = data.get('plot_filename')
            model_task = data.get('model_task', 'detection')  # Get model task parameter
            
            if not all([year, experiment, location, population, date, version_identifier, plot_filename]):
                return jsonify({'error': 'Missing required parameters'}), 400
            
            csv_path = None
            
            # Check if this is Plot_Images from split_orthomosaics
            if version_identifier == 'Plot_Images':
                # Look in Intermediate directory for Plot_Images results
                plot_images_dir = os.path.join(data_root_dir, 'Intermediate', year, experiment, location, population, 'plot_images', date)
                
                # Look for specific model task CSV first
                if model_task == 'segmentation':
                    primary_filename = f"{date}_Plot_Images_roboflow_predictions_segmentation.csv"
                else:
                    primary_filename = f"{date}_Plot_Images_roboflow_predictions_detection.csv"
                
                primary_path = os.path.join(plot_images_dir, primary_filename)
                if os.path.exists(primary_path):
                    csv_path = primary_path
                else:
                    # Fallback to generic filename if specific task file doesn't exist
                    fallback_filename = f"{date}_Plot_Images_roboflow_predictions.csv"
                    fallback_path = os.path.join(plot_images_dir, fallback_filename)
                    if os.path.exists(fallback_path):
                        csv_path = fallback_path
                
                # Last resort: search for any matching CSV but prefer the model task
                if not csv_path and os.path.exists(plot_images_dir):
                    preferred_suffix = f"_segmentation.csv" if model_task == 'segmentation' else f"_detection.csv"
                    for file in os.listdir(plot_images_dir):
                        if file.endswith(preferred_suffix) and "_Plot_Images_roboflow_predictions" in file:
                            csv_path = os.path.join(plot_images_dir, file)
                            break
                    
                    # If still not found, take any Plot_Images predictions file
                    if not csv_path:
                        for file in os.listdir(plot_images_dir):
                            if "_Plot_Images_roboflow_predictions" in file and file.endswith(".csv"):
                                csv_path = os.path.join(plot_images_dir, file)
                                break
            else:
                
                processed_dir = os.path.join(data_root_dir, 'Processed', year, experiment, location, population, date, platform, sensor)
                
                # Look for specific model task CSV first
                if model_task == 'segmentation':
                    primary_filename = f"{date}_{platform}_{sensor}_{version_identifier}_roboflow_predictions_segmentation.csv"
                else:
                    primary_filename = f"{date}_{platform}_{sensor}_{version_identifier}_roboflow_predictions_detection.csv"
                
                primary_path = os.path.join(processed_dir, primary_filename)
                if os.path.exists(primary_path):
                    csv_path = primary_path
                else:
                    # Fallback to generic filename if specific task file doesn't exist
                    fallback_filename = f"{date}_{platform}_{sensor}_{version_identifier}_roboflow_predictions.csv"
                    fallback_path = os.path.join(processed_dir, fallback_filename)
                    if os.path.exists(fallback_path):
                        csv_path = fallback_path
                
                # Last resort: search for any matching CSV but prefer the model task
                if not csv_path and os.path.exists(processed_dir):
                    preferred_suffix = f"_segmentation.csv" if model_task == 'segmentation' else f"_detection.csv"
                    for file in os.listdir(processed_dir):
                        if file.endswith(preferred_suffix) and f"{date}_{platform}_{sensor}_{version_identifier}_roboflow_predictions" in file:
                            csv_path = os.path.join(processed_dir, file)
                            break
                    
                    # If still not found, take any predictions file for this version
                    if not csv_path:
                        for file in os.listdir(processed_dir):
                            if file.startswith(f"{date}_{platform}_{sensor}_{version_identifier}_roboflow_predictions") and file.endswith(".csv"):
                                csv_path = os.path.join(processed_dir, file)
                                break
                            break
            
            if not csv_path or not os.path.exists(csv_path):
                return jsonify({'error': 'Inference results not found'}), 404
            
            df = pd.read_csv(csv_path)
            plot_predictions = df[df['image_file'] == plot_filename]
            predictions = []; class_counts = {}; has_segmentation=False; mask_count=0
            for _, row in plot_predictions.iterrows():
                pred = {'class': row['class'],'confidence': float(row['confidence']),'x': float(row['x']),'y': float(row['y']),'width': float(row['width']),'height': float(row['height']),'plot_index': int(row['plot_index']) if pd.notna(row['plot_index']) else None,'plot_label': row['plot_label'] if pd.notna(row['plot_label']) else None,'accession': row['accession'] if pd.notna(row['accession']) else None,'model_id': row.get('model_id') if pd.notna(row.get('model_id')) else None,'model_version': row.get('model_version') if pd.notna(row.get('model_version')) else None}
                # segmentation
                if 'points' in row and pd.notna(row['points']):
                    try:
                        pred['points'] = json.loads(row['points'])
                        has_segmentation = True; mask_count += 1
                    except Exception:
                        pass
                elif 'segments' in row and pd.notna(row['segments']):
                    try:
                        pred['segments'] = json.loads(row['segments'])
                        has_segmentation = True; mask_count += 1
                    except Exception:
                        pass
                predictions.append(pred)
                cn = row['class']; class_counts[cn] = class_counts.get(cn,0)+1
            return jsonify({'predictions': predictions,'class_counts': class_counts,'total_detections': len(predictions),'has_segmentation': has_segmentation,'mask_count': mask_count}), 200
        except Exception as e:
            print(f"Error getting plot predictions: {e}")
            return jsonify({'error': str(e)}), 500


def delete_inference_results_endpoint(file_app):
    @file_app.route('/delete_inference_results', methods=['POST'])
    def delete_inference_results():
        try:
            global inference_status
            data = request.json
            csv_path = data.get('csv_path')
            if not csv_path or not os.path.exists(csv_path):
                return jsonify({'error': 'CSV path not found'}), 400
            try:
                os.remove(csv_path)
            except Exception as e:
                return jsonify({'error': f'Failed to delete CSV: {e}'}), 500
            inference_status = {'running': False,'progress': 0,'message': '','error': None,'results': None,'completed': False}
            return jsonify({'message': 'Inference results deleted'}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500


def download_inference_csv_endpoint(file_app):
    """
    Download inference CSV file
    """
    @file_app.route('/download_inference_csv', methods=['POST'])
    def download_inference_csv():
        try:
            data = request.json
            csv_path = data.get('csv_path')
            
            if not csv_path or not os.path.exists(csv_path):
                return jsonify({'error': 'CSV file not found'}), 404
            
            # Extract filename for download
            filename = os.path.basename(csv_path)
            
            return send_file(
                csv_path,
                mimetype='text/csv',
                as_attachment=True,
                download_name=filename
            )
            
        except Exception as e:
            print(f"Error downloading inference CSV: {e}")
            return jsonify({'error': str(e)}), 500


def get_plot_boundary_info_endpoint(file_app, data_root_dir):
    """
    Get plot boundary information for filtering options
    """
    @file_app.route('/get_plot_boundary_info', methods=['POST'])
    def get_plot_boundary_info():
        try:
            data = request.json
            year = data.get('year')
            experiment = data.get('experiment')
            location = data.get('location')
            population = data.get('population')
            date = data.get('date')
            platform = data.get('platform')
            sensor = data.get('sensor')
            
            if not all([year, experiment, location, population]):
                return jsonify({'error': 'Missing required parameters (year, experiment, location, population)'}), 400
            
            plot_gdf = None
            source_file = None
            
            # First, try to find an existing traits GeoJSON file (if inference has been run before)
            if date and platform and sensor:
                plots_dir = os.path.join(
                    data_root_dir, 'Processed', year, experiment, location, population,
                    date, platform, sensor
                )
                
                if os.path.exists(plots_dir):
                    # Look for AgRowStitch directories
                    agrowstitch_dirs = [d for d in os.listdir(plots_dir) if d.startswith('AgRowStitch_v')]
                    
                    # Try to find an existing traits GeoJSON file
                    for agr_dir in sorted(agrowstitch_dirs, reverse=True):  # Try newest version first
                        traits_file = os.path.join(plots_dir, agr_dir, f"{date}-{platform}-{sensor}-{agr_dir}-Traits-WGS84.geojson")
                        if os.path.exists(traits_file):
                            try:
                                plot_gdf = gpd.read_file(traits_file)
                                source_file = traits_file
                                print(f"DEBUG: Found traits file for plot boundary info: {traits_file}")
                                break
                            except Exception as e:
                                print(f"DEBUG: Failed to read traits file {traits_file}: {e}")
                                continue
            
            # If no traits file found, try the standard plot boundary GeoJSON
            if plot_gdf is None:
                plot_boundary_geojson = os.path.join(
                    data_root_dir, 'Intermediate', year, experiment, location, population, 'Plot-Boundary-WGS84.geojson'
                )
                
                if os.path.exists(plot_boundary_geojson):
                    try:
                        plot_gdf = gpd.read_file(plot_boundary_geojson)
                        source_file = plot_boundary_geojson
                        print(f"DEBUG: Using plot boundary GeoJSON: {plot_boundary_geojson}")
                    except Exception as e:
                        print(f"DEBUG: Failed to read plot boundary file {plot_boundary_geojson}: {e}")
            
            # If still no data, try plot_borders.csv as fallback
            if plot_gdf is None:
                plot_borders_csv = os.path.join(
                    data_root_dir, "Raw", year, experiment, location, population, "plot_borders.csv"
                )
                
                if os.path.exists(plot_borders_csv):
                    try:
                        borders_df = pd.read_csv(plot_borders_csv)
                        print(f"DEBUG: Using plot borders CSV as fallback: {plot_borders_csv}")
                        print(f"DEBUG: CSV columns: {list(borders_df.columns)}")
                        
                        # Create a simple GeoDataFrame structure for extracting filter options
                        # Map common column variations
                        column_mapping = {
                            'Plot': 'plot',
                            'Accession': 'accession', 
                            'Population': 'population',
                            'Row': 'row',
                            'Column': 'column'
                        }
                        
                        # Normalize column names
                        for old_name, new_name in column_mapping.items():
                            if old_name in borders_df.columns and new_name not in borders_df.columns:
                                borders_df[new_name] = borders_df[old_name]
                        
                        # Add population if not present
                        if 'population' not in borders_df.columns:
                            borders_df['population'] = population
                        
                        plot_gdf = gpd.GeoDataFrame(borders_df)
                        source_file = plot_borders_csv
                        
                    except Exception as e:
                        print(f"DEBUG: Failed to read plot borders CSV {plot_borders_csv}: {e}")
            
            if plot_gdf is None:
                return jsonify({
                    'populations': [],
                    'columns': [],
                    'rows': [],
                    'total_plots': 0,
                    'message': 'No plot boundary data found. Please ensure plot_borders.csv exists in the Raw data directory.'
                }), 200
            
            # Extract unique values for filtering
            populations = []
            columns = []
            rows = []
            
            print(f"DEBUG: Plot data columns: {list(plot_gdf.columns)}")
            
            if 'population' in plot_gdf.columns:
                populations = sorted([str(x) for x in plot_gdf['population'].dropna().unique() if str(x) != 'nan'])
            
            if 'column' in plot_gdf.columns:
                columns = sorted([str(x) for x in plot_gdf['column'].dropna().unique() if str(x) != 'nan'])
            elif 'Column' in plot_gdf.columns:
                columns = sorted([str(x) for x in plot_gdf['Column'].dropna().unique() if str(x) != 'nan'])
            
            if 'row' in plot_gdf.columns:
                rows = sorted([str(x) for x in plot_gdf['row'].dropna().unique() if str(x) != 'nan'])
            elif 'Row' in plot_gdf.columns:
                rows = sorted([str(x) for x in plot_gdf['Row'].dropna().unique() if str(x) != 'nan'])
            
            print(f"DEBUG: Extracted filtering options - populations: {populations}, columns: {columns}, rows: {rows}")
            
            return jsonify({
                'populations': populations,
                'columns': columns,
                'rows': rows,
                'total_plots': len(plot_gdf),
                'source_file': source_file,
                'message': f'Plot boundary data loaded from {source_file.split("/")[-1] if source_file else "unknown"}'
            }), 200
            
        except Exception as e:
            print(f"Error getting plot boundary info: {e}")
            return jsonify({'error': str(e)}), 500


def update_traits_confidence_threshold_endpoint(file_app, data_root_dir):
    """
    Update trait counts in GeoJSON based on new confidence threshold
    """
    @file_app.route('/update_traits_confidence_threshold', methods=['POST'])
    def update_traits_confidence_threshold():
        try:
            data = request.json
            year = data.get('year')
            experiment = data.get('experiment') 
            location = data.get('location')
            population = data.get('population')
            date = data.get('date')
            platform = data.get('platform')
            sensor = data.get('sensor')
            agrowstitch_version = data.get('agrowstitch_version')
            orthomosaic = data.get('orthomosaic')
            confidence_threshold = float(data.get('confidence_threshold', 0.5))
            model_task = data.get('model_task', 'detection')
            
            version_identifier = orthomosaic if orthomosaic else agrowstitch_version
            
            if not all([year, experiment, location, population, date, version_identifier]):
                return jsonify({'error': 'Missing required parameters'}), 400
            
            # Find the predictions CSV file
            csv_path = None
            if version_identifier == 'Plot_Images':
                plot_images_dir = os.path.join(data_root_dir, 'Intermediate', year, experiment, location, population, 'plot_images', date)
                
                if model_task == 'segmentation':
                    primary_filename = f"{date}_Plot_Images_roboflow_predictions_segmentation.csv"
                else:
                    primary_filename = f"{date}_Plot_Images_roboflow_predictions_detection.csv"
                
                primary_path = os.path.join(plot_images_dir, primary_filename)
                if os.path.exists(primary_path):
                    csv_path = primary_path
                else:
                    fallback_filename = f"{date}_Plot_Images_roboflow_predictions.csv"
                    fallback_path = os.path.join(plot_images_dir, fallback_filename)
                    if os.path.exists(fallback_path):
                        csv_path = fallback_path
            else:
                processed_dir = os.path.join(data_root_dir, 'Processed', year, experiment, location, population, date, platform, sensor)
                
                if model_task == 'segmentation':
                    primary_filename = f"{date}_{platform}_{sensor}_{version_identifier}_roboflow_predictions_segmentation.csv"
                else:
                    primary_filename = f"{date}_{platform}_{sensor}_{version_identifier}_roboflow_predictions_detection.csv"
                
                primary_path = os.path.join(processed_dir, primary_filename)
                if os.path.exists(primary_path):
                    csv_path = primary_path
                else:
                    fallback_filename = f"{date}_{platform}_{sensor}_{version_identifier}_roboflow_predictions.csv"
                    fallback_path = os.path.join(processed_dir, fallback_filename)
                    if os.path.exists(fallback_path):
                        csv_path = fallback_path
            
            if not csv_path or not os.path.exists(csv_path):
                return jsonify({'error': 'Inference results CSV not found'}), 404
            
            # Find the traits GeoJSON file
            geojson_path = None
            if version_identifier == 'Plot_Images':
                output_dir = os.path.join(data_root_dir, 'Intermediate', year, experiment, location, population, 'plot_images', date)
                geojson_filename = f"{date}-Plot_Images-Traits-WGS84.geojson"
            else:
                output_dir = os.path.join(data_root_dir, 'Processed', year, experiment, location, population, date, platform, sensor, version_identifier)
                geojson_filename = f"{date}-{platform}-{sensor}-{version_identifier}-Traits-WGS84.geojson"
            
            geojson_path = os.path.join(output_dir, geojson_filename)
            
            if not os.path.exists(geojson_path):
                return jsonify({'error': 'Traits GeoJSON file not found'}), 404
            
            # Create backup of original GeoJSON if it doesn't exist
            backup_path = geojson_path.replace('.geojson', '_original_backup.geojson')
            if not os.path.exists(backup_path):
                import shutil
                shutil.copy2(geojson_path, backup_path)
            
            # Load predictions and filter by confidence threshold
            predictions_df = pd.read_csv(csv_path)
            filtered_predictions = predictions_df[predictions_df['confidence'] >= confidence_threshold]
            
            # Load existing traits GeoJSON
            traits_gdf = gpd.read_file(geojson_path)
            
            # Get model_id from predictions
            model_id = filtered_predictions['model_id'].iloc[0] if 'model_id' in filtered_predictions.columns and not filtered_predictions['model_id'].isna().all() else 'unknown'
            
            # Calculate new class counts based on filtered predictions
            if not filtered_predictions.empty:
                # Use the same normalization logic as create_traits_geojson
                def normalize_plot_label(val):
                    if pd.isna(val):
                        return None
                    sval = str(val).strip()
                    if sval.endswith('.0'):
                        sval = sval[:-2]
                    return sval
                
                # Group predictions by plot and class to get new counts
                class_counts_by_plot = {}
                
                for plot_label in filtered_predictions['plot_label'].unique():
                    if pd.isna(plot_label):
                        continue
                    
                    # Normalize the plot label for consistent matching
                    normalized_label = normalize_plot_label(plot_label)
                    if not normalized_label:
                        continue
                        
                    plot_preds = filtered_predictions[filtered_predictions['plot_label'] == plot_label]
                    class_counts_by_plot[normalized_label] = plot_preds['class'].value_counts().to_dict()
                
                # Update traits GeoJSON with new counts
                # First, reset all detection columns to 0
                detection_columns = [col for col in traits_gdf.columns if '/' in col and model_id in col]
                for col in detection_columns:
                    traits_gdf[col] = 0
                
                # Update with new counts
                for idx, row in traits_gdf.iterrows():
                    plot_val = normalize_plot_label(row.get('plot', ''))
                    
                    if plot_val and plot_val in class_counts_by_plot:
                        for class_name, count in class_counts_by_plot[plot_val].items():
                            col_name = f"{model_id}/{class_name}"
                            if col_name in traits_gdf.columns:
                                traits_gdf.loc[idx, col_name] = count
            else:
                # No predictions meet threshold - set all counts to 0
                detection_columns = [col for col in traits_gdf.columns if '/' in col and model_id in col]
                for col in detection_columns:
                    traits_gdf[col] = 0
            
            # Save updated GeoJSON
            traits_gdf.to_file(geojson_path, driver='GeoJSON')
            
            return jsonify({
                'success': True,
                'message': f'Trait counts updated for confidence threshold {confidence_threshold:.0%}',
                'updated_file': geojson_path
            }), 200
            
        except Exception as e:
            print(f"Error updating traits confidence threshold: {e}")
            return jsonify({'error': str(e)}), 500


def revert_traits_to_original_endpoint(file_app, data_root_dir):
    """
    Revert trait counts to original values before confidence threshold updates
    """
    @file_app.route('/revert_traits_to_original', methods=['POST'])
    def revert_traits_to_original():
        try:
            data = request.json
            year = data.get('year')
            experiment = data.get('experiment')
            location = data.get('location') 
            population = data.get('population')
            date = data.get('date')
            platform = data.get('platform')
            sensor = data.get('sensor')
            agrowstitch_version = data.get('agrowstitch_version')
            orthomosaic = data.get('orthomosaic')
            
            version_identifier = orthomosaic if orthomosaic else agrowstitch_version
            
            if not all([year, experiment, location, population, date, version_identifier]):
                return jsonify({'error': 'Missing required parameters'}), 400
            
            # Find the traits GeoJSON file and backup
            if version_identifier == 'Plot_Images':
                output_dir = os.path.join(data_root_dir, 'Intermediate', year, experiment, location, population, 'plot_images', date)
                geojson_filename = f"{date}-Plot_Images-Traits-WGS84.geojson"
            else:
                output_dir = os.path.join(data_root_dir, 'Processed', year, experiment, location, population, date, platform, sensor, version_identifier)
                geojson_filename = f"{date}-{platform}-{sensor}-{version_identifier}-Traits-WGS84.geojson"
            
            geojson_path = os.path.join(output_dir, geojson_filename)
            backup_path = geojson_path.replace('.geojson', '_original_backup.geojson')
            
            if not os.path.exists(backup_path):
                return jsonify({'error': 'Original backup file not found'}), 404
            
            # Restore from backup
            import shutil
            shutil.copy2(backup_path, geojson_path)
            
            return jsonify({
                'success': True,
                'message': 'Trait counts reverted to original values',
                'restored_file': geojson_path
            }), 200
            
        except Exception as e:
            print(f"Error reverting traits to original: {e}")
            return jsonify({'error': str(e)}), 500


def register_inference_routes(file_app, data_root_dir):
    run_roboflow_inference_endpoint(file_app, data_root_dir)
    get_inference_progress_endpoint(file_app)
    get_inference_results_endpoint(file_app, data_root_dir)
    get_plot_predictions_endpoint(file_app, data_root_dir)
    delete_inference_results_endpoint(file_app)
    download_inference_csv_endpoint(file_app)
    get_plot_boundary_info_endpoint(file_app, data_root_dir)
    update_traits_confidence_threshold_endpoint(file_app, data_root_dir)
    revert_traits_to_original_endpoint(file_app, data_root_dir)
