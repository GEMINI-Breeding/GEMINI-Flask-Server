"""
Roboflow Inference Module
Handles all Roboflow API inference operations for plot analysis
"""

import os
import threading
import base64
import re
import pandas as pd
import requests
from flask import jsonify, request


# Global variables for inference tracking
inference_status = {
    'running': False,
    'progress': 0,
    'message': '',
    'error': None,
    'results': None,
    'completed': False
}


def run_roboflow_inference_endpoint(file_app, data_root_dir):
    """
    Run Roboflow inference on stitched plot images
    """
    @file_app.route('/run_roboflow_inference', methods=['POST'])
    def run_roboflow_inference():
        global inference_status
        
        try:
            data = request.json
            api_url = data.get('apiUrl', 'https://infer.roboflow.com')
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
                
                # Load and encode image
                image_path = os.path.join(plots_dir, plot_file)
                with open(image_path, 'rb') as img_file:
                    img_data = img_file.read()
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                
                # Prepare Roboflow API request
                roboflow_url = f"{api_url}/infer/image"
                headers = {
                    'Content-Type': 'application/json'
                }
                
                payload = {
                    'api_key': api_key,
                    'model_id': model_id,
                    'image': {
                        'type': 'base64',
                        'value': img_base64
                    }
                }
                
                # Make inference request
                response = requests.post(roboflow_url, json=payload, headers=headers, timeout=60)
                
                if response.status_code == 200:
                    result = response.json()
                    predictions = result.get('predictions', [])
                    
                    # Process predictions
                    for pred in predictions:
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
                
                else:
                    print(f"Error in inference for {plot_file}: {response.status_code} - {response.text}")
                
                # Update progress
                progress = 30 + int((i + 1) / len(plot_files) * 60)
                inference_status['progress'] = progress
                inference_status['message'] = f'Processed {i + 1}/{len(plot_files)} plots'
                
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
            
            # Prepare results summary
            label_summary = [{'name': name, 'count': count} for name, count in label_counts.items()]
            
            inference_status['results'] = {
                'csvPath': csv_path,
                'totalPlots': len(plot_files),
                'totalPredictions': len(all_predictions),
                'labels': label_summary
            }
        else:
            inference_status['results'] = {
                'csvPath': None,
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
