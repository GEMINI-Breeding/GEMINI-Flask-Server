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
from flask import jsonify, request
from inference_sdk import InferenceHTTPClient
from shapely.geometry import Polygon

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


def create_traits_geojson(predictions_df, data_root_dir, year, experiment, location, population, date, platform, sensor, agrowstitch_dir):
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
        print(f"DEBUG: Starting create_traits_geojson with parameters:")
        print(f"  data_root_dir: {data_root_dir}")
        print(f"  year: {year}, experiment: {experiment}, location: {location}")
        print(f"  population: {population}, date: {date}, platform: {platform}, sensor: {sensor}")
        print(f"  agrowstitch_dir: {agrowstitch_dir}")
        print(f"  predictions_df shape: {predictions_df.shape}")
        
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
        output_dir = os.path.join(
            data_root_dir, 'Processed', year, experiment, location, population,
            date, platform, sensor, agrowstitch_dir
        )
        
        print(f"DEBUG: Creating output directory at orthomosaic level: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Verify directory was created
        if os.path.exists(output_dir):
            print(f"Output directory exists: {output_dir}")
        else:
            print(f"ERROR: Failed to create output directory: {output_dir}")
        
        # Include agrowstitch_dir in the filename to make it unique per version
        geojson_filename = f"{date}-{platform}-{sensor}-{agrowstitch_dir}-Traits-WGS84.geojson"
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
            # Support for local or cloud inference
            inference_mode = data.get('inferenceMode', 'cloud')  # 'cloud' or 'local'
            import subprocess
            import socket
            def is_local_inference_running(host='localhost', port=9001):
                try:
                    with socket.create_connection((host, port), timeout=2):
                        return True
                except Exception:
                    return False

            if inference_mode == 'local':
                api_url = data.get('apiUrl', 'http://localhost:9001')
                # Check if local inference server is running, if not, start it
                if not is_local_inference_running():
                    try:
                        # Start the inference server in the background (verbose output)
                        subprocess.Popen([
                            'inference', 'server', 'start'
                        ])
                        # Wait for the server to be ready
                        max_wait = 120
                        waited = 0
                        while not is_local_inference_running() and waited < max_wait:
                            time.sleep(1)
                            waited += 1
                        if not is_local_inference_running():
                            return jsonify({'error': 'Local inference server failed to start within 2 minutes.'}), 500
                    except Exception as e:
                        return jsonify({'error': f'Failed to start local inference server: {e}'}), 500
            else:
                api_url = data.get('apiUrl', 'https://detect.roboflow.com')
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
            
            if not all([api_key, model_id, year, experiment, location, population, date, platform, sensor, agrowstitch_dir]):
                return jsonify({'error': 'Missing required parameters'}), 400
                
            # Reset inference status
            inference_status = {
                'running': True,
                'progress': 0,
                'message': 'Initializing inference...',
                'error': None,
                'results': None,
                'completed': False
            }
            
            # Start inference in background thread
            thread = threading.Thread(
                target=run_roboflow_inference_worker,
                args=(api_url, api_key, model_id, year, experiment, location, population, date, platform, sensor, agrowstitch_dir, data_root_dir),
                daemon=True
            )
            thread.start()
            
            return jsonify({'message': 'Inference started successfully'}), 200
            
        except Exception as e:
            print(f"Error starting Roboflow inference: {e}")
            inference_status['error'] = str(e)
            return jsonify({'error': str(e)}), 500


def run_roboflow_inference_worker(api_url, api_key, model_id, year, experiment, location, population, date, platform, sensor, agrowstitch_dir, data_root_dir):
    """
    Worker function to run Roboflow inference in background
    """
    global inference_status
    
    try:
        inference_status['message'] = 'Loading plot images...'
        inference_status['progress'] = 10
        
        # Build path to AgRowStitch plot images
        plots_dir = os.path.join(
            data_root_dir, 'Processed', year, experiment, location, population,
            date, platform, sensor, agrowstitch_dir
        )
        
        if not os.path.exists(plots_dir):
            inference_status['error'] = f'AgRowStitch directory not found: {plots_dir}'
            inference_status['running'] = False
            return
            
        # Find all plot image files
        plot_files = [f for f in os.listdir(plots_dir) 
                     if f.startswith('full_res_mosaic_temp_plot_') and f.endswith('.png')]
        
        if not plot_files:
            inference_status['error'] = 'No plot images found in AgRowStitch directory'
            inference_status['running'] = False
            return
            
        inference_status['message'] = f'Found {len(plot_files)} plot images'
        inference_status['progress'] = 20
        
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
        
        # Initialize Roboflow client
        inference_status['message'] = 'Initializing Roboflow client...'
        inference_status['progress'] = 25
        
        try:
            client = InferenceHTTPClient(
                api_url=api_url,
                api_key=api_key
            )
        except Exception as e:
            inference_status['error'] = f'Failed to initialize Roboflow client: {e}'
            inference_status['running'] = False
            return
        
        inference_status['message'] = 'Running inference on plots...'
        inference_status['progress'] = 30
        
        for i, plot_file in enumerate(sorted(plot_files)):
            try:
                # Extract plot index from filename
                plot_match = re.search(r'temp_plot_(\d+)', plot_file)
                if not plot_match:
                    continue
                    
                plot_index = int(plot_match.group(1))
                plot_info = plot_data.get(plot_index, {})
                
                # Get full image path
                image_path = os.path.join(plots_dir, plot_file)
                
                inference_status['message'] = f'Processing plot {plot_index} ({i + 1}/{len(plot_files)}) - cropping image...'
                
                # Crop the image into patches with overlap
                crops = crop_image_with_overlap(image_path, crop_size=640, overlap=32)

                if not crops:
                    print(f"Warning: No crops generated for {plot_file}")
                    continue
                
                plot_predictions = []
                temp_dirs_to_cleanup = set()
                
                # Process each crop
                for j, crop_info in enumerate(crops):
                    try:
                        inference_status['message'] = f'Processing plot {plot_index} crop {j + 1}/{len(crops)}'
                        
                        # Track temp directory for cleanup
                        temp_dirs_to_cleanup.add(crop_info['temp_dir'])
                        
                        # Use Roboflow SDK for inference
                        result = client.infer(crop_info['crop_path'], model_id=model_id)
                        
                        if 'predictions' in result:
                            crop_predictions = result['predictions']
                            
                            # Extract model version from result if available
                            model_version = None
                            if hasattr(result, 'model') and hasattr(result.model, 'version'):
                                model_version = result.model.version
                            elif isinstance(result, dict) and 'model' in result:
                                model_info = result['model']
                                if isinstance(model_info, dict) and 'version' in model_info:
                                    model_version = model_info['version']
                            
                            # Add model information to each prediction
                            for pred in crop_predictions:
                                pred['model_id'] = model_id
                                pred['model_version'] = model_version
                            
                            # Transform predictions to plot coordinates
                            transformed_predictions = transform_predictions_to_plot_coordinates(
                                crop_predictions, crop_info
                            )
                            
                            plot_predictions.extend(transformed_predictions)
                            
                        else:
                            print(f"No predictions found for {plot_file} crop {j}")
                    
                    except Exception as e:
                        print(f"Error processing crop {j} of {plot_file}: {e}")
                        continue
                
                # Clean up temporary crop files
                for temp_dir in temp_dirs_to_cleanup:
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception as e:
                        print(f"Warning: Could not clean up temp directory {temp_dir}: {e}")
                
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
                    all_predictions.append(prediction_data)
                    
                    # Count labels
                    class_name = pred.get('class', 'unknown')
                    label_counts[class_name] = label_counts.get(class_name, 0) + 1
                
                print(f"Processed plot {plot_index}: {len(crops)} crops -> {len(plot_predictions)} raw predictions -> {len(final_predictions)} final predictions")
                
                # Update progress
                progress = 30 + int((i + 1) / len(plot_files) * 60)
                inference_status['progress'] = progress
                inference_status['message'] = f'Completed plot {plot_index} ({i + 1}/{len(plot_files)})'
                
            except Exception as e:
                print(f"Error processing {plot_file}: {e}")
                continue
        
        # Save results to CSV
        inference_status['message'] = 'Saving results...'
        inference_status['progress'] = 95
        
        if all_predictions:
            predictions_df = pd.DataFrame(all_predictions)
            
            # Create output directory
            output_dir = os.path.join(
                data_root_dir, 'Processed', year, experiment, location, population,
                date, platform, sensor
            )
            os.makedirs(output_dir, exist_ok=True)
            
            # Save CSV
            csv_filename = f"{date}_{platform}_{sensor}_{agrowstitch_dir}_roboflow_predictions.csv"
            csv_path = os.path.join(output_dir, csv_filename)
            predictions_df.to_csv(csv_path, index=False)
            
            # Create traits GeoJSON for integration with stats/map tabs
            inference_status['message'] = 'Creating traits GeoJSON...'
            geojson_path = create_traits_geojson(
                predictions_df, data_root_dir, year, experiment, location, 
                population, date, platform, sensor, agrowstitch_dir
            )
            
            # Prepare results summary
            label_summary = [{'name': name, 'count': count} for name, count in label_counts.items()]
            
            inference_status['results'] = {
                'csvPath': csv_path,
                'geojsonPath': geojson_path,
                'totalPlots': len(plot_files),
                'totalPredictions': len(all_predictions),
                'labels': label_summary
            }
        else:
            inference_status['results'] = {
                'csvPath': None,
                'geojsonPath': None,
                'totalPlots': len(plot_files),
                'totalPredictions': 0,
                'labels': []
            }
        
        inference_status['message'] = 'Inference completed successfully!'
        inference_status['progress'] = 100
        inference_status['completed'] = True
        inference_status['running'] = False
        
    except Exception as e:
        print(f"Error in Roboflow inference worker: {e}")
        inference_status['error'] = str(e)
        inference_status['running'] = False


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
            
            # Check processed directory for inference results
            dates_dir = os.path.join(data_root_dir, 'Processed', year, experiment, location, population)
            
            if not os.path.exists(dates_dir):
                return jsonify({'results': results}), 200
                
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
                        
                        # Look for inference CSV files
                        for file in os.listdir(sensor_path):
                            if file.endswith('_roboflow_predictions.csv'):
                                csv_path = os.path.join(sensor_path, file)
                                
                                # Extract agrowstitch version from filename
                                # Format: date_platform_sensor_agrowstitchversion_roboflow_predictions.csv
                                parts = file.split('_')
                                if len(parts) >= 5:
                                    orthomosaic_version = '_'.join(parts[3:-2])  # Everything between sensor and 'roboflow'
                                else:
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
                                    'timestamp': os.path.getmtime(csv_path)
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
            year = data.get('year')
            experiment = data.get('experiment')
            location = data.get('location')
            population = data.get('population')
            date = data.get('date')
            platform = data.get('platform')
            sensor = data.get('sensor')
            agrowstitch_version = data.get('agrowstitch_version')  # Keep for backward compatibility
            orthomosaic = data.get('orthomosaic')  # New parameter name
            
            # Use orthomosaic if provided, otherwise fall back to agrowstitch_version
            version_identifier = orthomosaic if orthomosaic else agrowstitch_version
            
            plot_filename = data.get('plot_filename')
            
            if not all([year, experiment, location, population, date, platform, sensor, version_identifier, plot_filename]):
                return jsonify({'error': 'Missing required parameters'}), 400
                
            # Find the CSV file
            csv_filename = f"{date}_{platform}_{sensor}_{version_identifier}_roboflow_predictions.csv"
            csv_path = os.path.join(
                data_root_dir, 'Processed', year, experiment, location, population,
                date, platform, sensor, csv_filename
            )
            
            if not os.path.exists(csv_path):
                return jsonify({'error': 'Inference results not found'}), 404
                
            # Load predictions for this specific plot
            df = pd.read_csv(csv_path)
            
            # Filter by image file
            plot_predictions = df[df['image_file'] == plot_filename]
            
            # Convert to list of dictionaries
            predictions = []
            class_counts = {}
            
            for _, row in plot_predictions.iterrows():
                prediction = {
                    'class': row['class'],
                    'confidence': float(row['confidence']),
                    'x': float(row['x']),
                    'y': float(row['y']),
                    'width': float(row['width']),
                    'height': float(row['height']),
                    'plot_index': int(row['plot_index']) if pd.notna(row['plot_index']) else None,
                    'plot_label': row['plot_label'] if pd.notna(row['plot_label']) else None,
                    'accession': row['accession'] if pd.notna(row['accession']) else None,
                    'model_id': row.get('model_id') if pd.notna(row.get('model_id')) else None,
                    'model_version': row.get('model_version') if pd.notna(row.get('model_version')) else None
                }
                predictions.append(prediction)
                
                # Count classes
                class_name = row['class']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            return jsonify({
                'predictions': predictions,
                'class_counts': class_counts,
                'total_detections': len(predictions)
            }), 200
            
        except Exception as e:
            print(f"Error getting plot predictions: {e}")
            return jsonify({'error': str(e)}), 500


def delete_inference_results_endpoint(file_app, data_root_dir):
    """
    Delete inference results including CSV file and optionally the GeoJSON traits file
    """
    @file_app.route('/delete_inference_results', methods=['POST'])
    def delete_inference_results():
        try:
            data = request.json
            year = data.get('year')
            experiment = data.get('experiment')
            location = data.get('location')
            population = data.get('population')
            date = data.get('date')
            platform = data.get('platform')
            sensor = data.get('sensor')
            orthomosaic = data.get('orthomosaic')
            delete_traits = data.get('delete_traits', False)  # Optional: also delete traits GeoJSON
            
            if not all([year, experiment, location, population, date, platform, sensor, orthomosaic]):
                return jsonify({'error': 'Missing required parameters'}), 400
            
            # Build paths
            output_dir = os.path.join(
                data_root_dir, 'Processed', year, experiment, location, population,
                date, platform, sensor
            )
            
            csv_filename = f"{date}_{platform}_{sensor}_{orthomosaic}_roboflow_predictions.csv"
            csv_path = os.path.join(output_dir, csv_filename)
            
            deleted_files = []
            
            # Delete CSV file
            if os.path.exists(csv_path):
                os.remove(csv_path)
                deleted_files.append(csv_filename)
                print(f"Deleted inference CSV: {csv_path}")
            else:
                return jsonify({'error': 'Inference results file not found'}), 404
            
            # Optionally delete traits GeoJSON file
            if delete_traits:
                # Look for GeoJSON in the orthomosaic version directory with updated filename
                geojson_filename = f"{date}-{platform}-{sensor}-{orthomosaic}-Traits-WGS84.geojson"
                geojson_dir = os.path.join(output_dir, orthomosaic)
                geojson_path = os.path.join(geojson_dir, geojson_filename)
                
                if os.path.exists(geojson_path):
                    # Check if the GeoJSON file contains inference data (detection counts)
                    try:
                        gdf = gpd.read_file(geojson_path)
                        
                        # Check if it has detection-related columns (new naming scheme)
                        detection_columns = [col for col in gdf.columns if 
                                           'Total_Detections_' in col or '/' in col]
                        
                        if detection_columns:
                            # Remove only the detection columns, keep other traits
                            gdf_cleaned = gdf.drop(columns=detection_columns)
                            
                            # If there are still other trait columns, save the cleaned version
                            trait_columns = [col for col in gdf_cleaned.columns 
                                           if col not in ['geometry', 'plot_index', 'Plot', 'Bed', 'Tier', 'accession', 'population']]
                            
                            if trait_columns:
                                gdf_cleaned.to_file(geojson_path, driver='GeoJSON')
                                deleted_files.append(f"Detection data ({len(detection_columns)} columns) from {geojson_filename}")
                            else:
                                # No other traits, delete the entire file
                                os.remove(geojson_path)
                                deleted_files.append(geojson_filename)
                                deleted_files.append(geojson_filename)
                        else:
                            print(f"GeoJSON file {geojson_path} does not contain detection data")
                            
                    except Exception as e:
                        print(f"Error processing GeoJSON file: {e}")
                        # If we can't process it, just delete it if explicitly requested
                        os.remove(geojson_path)
                        deleted_files.append(geojson_filename)
            
            return jsonify({
                'message': 'Inference results deleted successfully',
                'deleted_files': deleted_files
            }), 200
            
        except Exception as e:
            print(f"Error deleting inference results: {e}")
            return jsonify({'error': str(e)}), 500


def register_inference_routes(file_app, data_root_dir):
    """
    Register all inference-related routes with the Flask app
    """
    run_roboflow_inference_endpoint(file_app, data_root_dir)
    get_inference_progress_endpoint(file_app)
    get_agrowstitch_versions_endpoint(file_app, data_root_dir)
    get_inference_results_endpoint(file_app, data_root_dir)
    get_plot_predictions_endpoint(file_app, data_root_dir)
    delete_inference_results_endpoint(file_app, data_root_dir)
