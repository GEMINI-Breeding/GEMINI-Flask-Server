"""
Roboflow Inference Module
Handles all Roboflow API inference operations for plot analysis
"""

import os
import threading
import re
import tempfile
import shutil
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
            'crop_id': crop_info['crop_id']
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
        # Look for existing plot boundary GeoJSON first
        plot_boundary_geojson = os.path.join(
            data_root_dir, 'Intermediate', year, experiment, location, population, 'Plot-Boundary-WGS84.geojson'
        )
        
        # If no plot boundary GeoJSON, try to create from plot_borders.csv
        if not os.path.exists(plot_boundary_geojson):
            plot_borders_csv = os.path.join(
                data_root_dir, "Raw", year, experiment, location, population, "plot_borders.csv"
            )
            
            if not os.path.exists(plot_borders_csv):
                print("Warning: No plot boundaries found for traits export")
                return None
            
            # Create simple plot boundaries from CSV (this is a simplified approach)
            borders_df = pd.read_csv(plot_borders_csv)
            
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
                'Bed': borders_df.get('Bed', 1),
                'Tier': borders_df.get('Tier', 1),
                'accession': borders_df.get('Accession', 'Unknown'),
                'population': population,
                'geometry': geometries
            }, crs='EPSG:4326')
        else:
            # Load existing plot boundary GeoJSON
            plot_gdf = gpd.read_file(plot_boundary_geojson)
        
        # Aggregate predictions by plot
        if not predictions_df.empty:
            # Calculate total detections per plot
            total_detections = predictions_df.groupby('plot_index').size()
            
            # Create summary columns for each detected class
            class_columns = {}
            for class_name in predictions_df['class'].unique():
                if pd.notna(class_name):
                    class_counts = predictions_df[predictions_df['class'] == class_name].groupby('plot_index').size()
                    class_columns[f'Total {class_name.title()}'] = class_counts
            
            # Merge with plot boundaries
            traits_gdf = plot_gdf.copy()
            
            # Add detection counts to the GeoDataFrame
            for plot_idx in traits_gdf['plot_index']:
                if plot_idx in total_detections.index:
                    traits_gdf.loc[traits_gdf['plot_index'] == plot_idx, 'Total Detections'] = int(total_detections[plot_idx])
                else:
                    traits_gdf.loc[traits_gdf['plot_index'] == plot_idx, 'Total Detections'] = 0
                
                # Add individual class counts
                for class_col, class_series in class_columns.items():
                    if plot_idx in class_series.index:
                        traits_gdf.loc[traits_gdf['plot_index'] == plot_idx, class_col] = int(class_series[plot_idx])
                    else:
                        traits_gdf.loc[traits_gdf['plot_index'] == plot_idx, class_col] = 0
        else:
            # No predictions found, just add zero counts
            traits_gdf = plot_gdf.copy()
            traits_gdf['Total Detections'] = 0
        
        # Ensure standard columns exist
        required_columns = ['Bed', 'Tier', 'Plot', 'accession', 'population']
        for col in required_columns:
            if col not in traits_gdf.columns:
                if col == 'Plot':
                    traits_gdf[col] = traits_gdf['plot_index']
                elif col == 'Bed':
                    traits_gdf[col] = 1
                elif col == 'Tier':
                    traits_gdf[col] = 1
                elif col == 'accession':
                    traits_gdf[col] = 'Unknown'
                elif col == 'population':
                    traits_gdf[col] = population
        
        # Convert data types
        traits_gdf['Bed'] = traits_gdf['Bed'].astype(int)
        traits_gdf['Tier'] = traits_gdf['Tier'].astype(int)
        traits_gdf['Total Detections'] = traits_gdf['Total Detections'].astype(int)
        
        # Save the traits GeoJSON file
        output_dir = os.path.join(
            data_root_dir, 'Processed', year, experiment, location, population,
            date, platform, sensor
        )
        os.makedirs(output_dir, exist_ok=True)
        
        geojson_filename = f"{date}-{platform}-{sensor}-Traits-WGS84.geojson"
        geojson_path = os.path.join(output_dir, geojson_filename)
        
        # Check if traits file already exists and merge if so
        if os.path.exists(geojson_path):
            try:
                existing_gdf = gpd.read_file(geojson_path)
                
                # Merge on common columns, keeping all existing traits and adding new detection data
                merge_columns = ['Bed', 'Tier', 'Plot', 'accession', 'population', 'geometry']
                
                # Only merge on columns that exist in both dataframes
                available_merge_cols = [col for col in merge_columns if col in existing_gdf.columns and col in traits_gdf.columns]
                
                if available_merge_cols:
                    # Update existing records with new detection data
                    for idx, row in traits_gdf.iterrows():
                        # Find matching row in existing data
                        plot_idx = row.get('plot_index', row.get('Plot'))
                        mask = existing_gdf['Plot'] == plot_idx
                        
                        if mask.any():
                            # Update existing row with new detection data
                            for col in traits_gdf.columns:
                                if col.startswith('Total ') and col not in existing_gdf.columns:
                                    existing_gdf.loc[mask, col] = row[col]
                                elif col.startswith('Total '):
                                    existing_gdf.loc[mask, col] = row[col]
                    
                    traits_gdf = existing_gdf
                
            except Exception as e:
                print(f"Warning: Could not merge with existing traits file: {e}")
        
        # Save the final GeoJSON
        traits_gdf.to_file(geojson_path, driver='GeoJSON')
        print(f"Saved traits GeoJSON: {geojson_path}")
        
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
                        'height': pred.get('height', 0)
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
                                    agrowstitch_version = '_'.join(parts[3:-2])  # Everything between sensor and 'roboflow'
                                else:
                                    agrowstitch_version = 'unknown'
                                
                                # Check if corresponding plot images exist
                                agrowstitch_dir = os.path.join(sensor_path, agrowstitch_version)
                                plot_images_exist = False
                                plot_count = 0
                                
                                if os.path.exists(agrowstitch_dir):
                                    plot_files = [f for f in os.listdir(agrowstitch_dir) 
                                                if f.startswith('full_res_mosaic_temp_plot_') and f.endswith('.png')]
                                    plot_images_exist = len(plot_files) > 0
                                    plot_count = len(plot_files)
                                
                                # Get basic statistics from CSV
                                total_predictions = 0
                                classes_detected = []
                                
                                try:
                                    df = pd.read_csv(csv_path)
                                    total_predictions = len(df)
                                    classes_detected = df['class'].value_counts().to_dict()
                                except Exception as e:
                                    print(f"Error reading CSV {csv_path}: {e}")
                                
                                result_entry = {
                                    'id': f"{date}-{platform}-{sensor}-{agrowstitch_version}",
                                    'date': date,
                                    'platform': platform,
                                    'sensor': sensor,
                                    'agrowstitch_version': agrowstitch_version,
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
            agrowstitch_version = data.get('agrowstitch_version')
            plot_filename = data.get('plot_filename')
            
            if not all([year, experiment, location, population, date, platform, sensor, agrowstitch_version, plot_filename]):
                return jsonify({'error': 'Missing required parameters'}), 400
                
            # Find the CSV file
            csv_filename = f"{date}_{platform}_{sensor}_{agrowstitch_version}_roboflow_predictions.csv"
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
                    'accession': row['accession'] if pd.notna(row['accession']) else None
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


def register_inference_routes(file_app, data_root_dir):
    """
    Register all inference-related routes with the Flask app
    """
    run_roboflow_inference_endpoint(file_app, data_root_dir)
    get_inference_progress_endpoint(file_app)
    get_agrowstitch_versions_endpoint(file_app, data_root_dir)
    get_inference_results_endpoint(file_app, data_root_dir)
    get_plot_predictions_endpoint(file_app, data_root_dir)
