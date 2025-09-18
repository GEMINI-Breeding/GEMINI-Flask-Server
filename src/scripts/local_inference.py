"""
Local Model Inference Module
Handles inference operations using locally stored PyTorch models
"""
import os
import tempfile
import json
import threading
import glob
import re
import shutil
import traceback
import pandas as pd
import geopandas as gpd
from pathlib import Path
from PIL import Image
import torch
import cv2
import numpy as np
from flask import jsonify, request, send_file
from shapely.geometry import Polygon, Point

# Try to import ultralytics for YOLOv8/YOLOv11 support
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
    print("Ultralytics library available - YOLOv8/v11 models supported")
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Ultralytics library not available - using torch.load for model loading")

# Global variables for inference tracking
local_inference_status = {
    'running': False,
    'progress': 0,
    'message': '',
    'error': None,
    'results': None,
    'completed': False
}


def load_local_model(model_path, device='cpu'):
    """
    Load a local PyTorch model, prioritizing metadata file for task type.
    
    Args:
        model_path (str): Path to the model file (.pt)
        device (str): Device to load model on ('cpu' or 'cuda')
    
    Returns:
        tuple: (model, model_type) where model_type is 'detection' or 'segmentation'
    """
    try:
        model_dir = os.path.dirname(model_path)
        metadata_path = os.path.join(model_dir, 'model_metadata.json')
        model_type = 'detection' # Default value

        # 1. Prioritize reading from metadata file
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                model_type = metadata.get('task', 'detection')
                print(f"Loaded model type '{model_type}' from metadata file.")
            except Exception as e:
                print(f"Warning: Could not read metadata file {metadata_path}: {e}")

        # 2. Load the model using the appropriate library
        model = None
        if ULTRALYTICS_AVAILABLE:
            try:
                model = YOLO(model_path)
                # If metadata was not found, inspect the model now
                if not os.path.exists(metadata_path):
                    if hasattr(model, 'task'):
                        model_type = 'segmentation' if model.task == 'segment' else 'detection'
                    print(f"Inspected model task from Ultralytics model: {model_type}")
            except Exception as e:
                print(f"Failed to load with ultralytics: {e}, trying torch.load...")
                model = torch.load(model_path, map_location=device)
        else:
            model = torch.load(model_path, map_location=device)

        print(f"Successfully loaded model: {os.path.basename(model_path)}, Type: {model_type}")
        return model, model_type
        
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        raise e


def run_inference_on_image(model, image_path, model_type='detection', confidence=0.25):
    """
    Run inference on a single image using the local model
    
    Args:
        model: Loaded PyTorch model
        image_path (str): Path to image file
        model_type (str): 'detection' or 'segmentation'
        confidence (float): Confidence threshold
    
    Returns:
        list: List of predictions with coordinates and class information
    """
    try:
        predictions = []
        
        if ULTRALYTICS_AVAILABLE and hasattr(model, 'predict') and str(type(model).__name__) == 'YOLO':
            results = model.predict(image_path, conf=confidence, verbose=False)
            
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes.data.cpu().numpy()
                    
                    for box_idx, box in enumerate(boxes):
                        x1, y1, x2, y2, conf, cls = box[:6]
                        x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2
                        width, height = x2 - x1, y2 - y1
                        class_name = model.names[int(cls)] if hasattr(model, 'names') else str(int(cls))
                        
                        pred = {
                            'class': class_name, 'confidence': float(conf), 'x': float(x_center),
                            'y': float(y_center), 'width': float(width), 'height': float(height)
                        }
                        
                        if model_type == 'segmentation' and hasattr(result, 'masks') and result.masks is not None:
                            try:
                                if box_idx < len(result.masks.data):
                                    mask = result.masks.data[box_idx].cpu().numpy()
                                    contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    if contours:
                                        largest_contour = max(contours, key=cv2.contourArea)
                                        points = [{'x': int(point[0][0]), 'y': int(point[0][1])} for point in largest_contour]
                                        pred['points'] = points
                            except Exception as e:
                                print(f"Error processing segmentation mask: {e}")
                        
                        predictions.append(pred)
            return predictions
        
        elif hasattr(model, 'predict'):
            # Fallback for other PyTorch models with a predict method
            # This logic may need to be adapted for non-YOLO models
            # For now, it mirrors the YOLO logic
            results = model.predict(image_path, conf=confidence, verbose=False)
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes.data.cpu().numpy()
                    for box_idx, box in enumerate(boxes):
                        x1, y1, x2, y2, conf, cls = box[:6]
                        pred = {
                            'class': model.names[int(cls)], 'confidence': float(conf),
                            'x': (x1 + x2) / 2, 'y': (y1 + y2) / 2, 'width': x2 - x1, 'height': y2 - y1
                        }
                        if model_type == 'segmentation' and hasattr(result, 'masks') and result.masks is not None:
                            if box_idx < len(result.masks.data):
                                mask_data = result.masks.data.cpu().numpy()[box_idx]
                                contours, _ = cv2.findContours((mask_data * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                if contours:
                                    largest_contour = max(contours, key=cv2.contourArea)
                                    points = [{'x': int(p[0][0]), 'y': int(p[0][1])} for p in largest_contour]
                                    pred['points'] = points
                        predictions.append(pred)
            return predictions
            
        else:
            print(f"Model type not supported for inference: {type(model)}")
            return []
            
    except Exception as e:
        print(f"Error running inference on {image_path}: {e}")
        return []

def crop_image_with_overlap(image_path, crop_size=640, overlap=32):
    """Crops a large image into smaller patches with overlap."""
    crops, temp_dir = [], None
    try:
        image = Image.open(image_path)
        img_width, img_height = image.size
        stride = crop_size - overlap
        
        y_positions = list(range(0, img_height - crop_size + 1, stride))
        if not y_positions or y_positions[-1] + crop_size < img_height:
            y_positions.append(img_height - crop_size if img_height > crop_size else 0)
        
        x_positions = list(range(0, img_width - crop_size + 1, stride))
        if not x_positions or x_positions[-1] + crop_size < img_width:
            x_positions.append(img_width - crop_size if img_width > crop_size else 0)

        temp_dir = tempfile.mkdtemp()
        crop_id = 0
        for y in set(y_positions):
            for x in set(x_positions):
                actual_x = min(x, img_width - crop_size) if img_width > crop_size else 0
                actual_y = min(y, img_height - crop_size) if img_height > crop_size else 0
                box = (actual_x, actual_y, actual_x + crop_size, actual_y + crop_size)
                crop = image.crop(box)
                crop_path = os.path.join(temp_dir, f"crop_{crop_id}.jpg")
                crop.save(crop_path, format='JPEG', quality=95)
                crops.append({'crop_id': crop_id, 'x_offset': actual_x, 'y_offset': actual_y, 'crop_path': crop_path, 'temp_dir': temp_dir})
                crop_id += 1
        return crops
    except Exception as e:
        print(f"Error cropping image {image_path}: {e}")
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return []

def transform_predictions_to_plot_coordinates(predictions, crop_info):
    """Transforms prediction coordinates from crop level to plot level."""
    if predictions is None: return []
    transformed = []
    for pred in predictions:
        transformed_pred = {
            'class': pred.get('class', ''), 'confidence': pred.get('confidence', 0),
            'x': pred.get('x', 0) + crop_info['x_offset'], 'y': pred.get('y', 0) + crop_info['y_offset'],
            'width': pred.get('width', 0), 'height': pred.get('height', 0), 'crop_id': crop_info['crop_id']
        }
        if 'points' in pred:
            transformed_pred['points'] = [{'x': p['x'] + crop_info['x_offset'], 'y': p['y'] + crop_info['y_offset']} for p in pred['points']]
        transformed.append(transformed_pred)
    return transformed

def apply_nms(predictions, iou_threshold=0.5):
    """Applies Non-Maximum Suppression to remove duplicate detections."""
    if not predictions: return []
    class_groups = {}
    for pred in predictions:
        class_groups.setdefault(pred['class'], []).append(pred)
    
    final_predictions = []
    for class_name, class_predictions in class_groups.items():
        class_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        keep = []
        while class_predictions:
            current = class_predictions.pop(0)
            keep.append(current)
            class_predictions = [p for p in class_predictions if calculate_iou(current, p) < iou_threshold]
        final_predictions.extend(keep)
    return final_predictions

def calculate_iou(box1, box2):
    """Calculates Intersection over Union for two bounding boxes."""
    x1_min, y1_min = box1['x'] - box1['width'] / 2, box1['y'] - box1['height'] / 2
    x1_max, y1_max = box1['x'] + box1['width'] / 2, box1['y'] + box1['height'] / 2
    x2_min, y2_min = box2['x'] - box2['width'] / 2, box2['y'] - box2['height'] / 2
    x2_max, y2_max = box2['x'] + box2['width'] / 2, box2['y'] + box2['height'] / 2
    
    inter_x_min, inter_y_min = max(x1_min, x2_min), max(y1_min, y2_min)
    inter_x_max, inter_y_max = min(x1_max, x2_max), min(y1_max, y2_max)
    
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area, box2_area = box1['width'] * box1['height'], box2['width'] * box2['height']
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def create_traits_geojson(predictions_df, data_root_dir, year, experiment, location, population, date, platform, sensor, agrowstitch_dir):
    """Creates a GeoJSON file with detection counts per plot."""
    try:
        plot_boundary_geojson = os.path.join(data_root_dir, 'Intermediate', year, experiment, location, population, 'Plot-Boundary-WGS84.geojson')
        if not os.path.exists(plot_boundary_geojson):
            print(f"ERROR: Plot boundary GeoJSON not found at {plot_boundary_geojson}")
            return None
        
        plot_gdf = gpd.read_file(plot_boundary_geojson)
        plot_counts = predictions_df.groupby(['plot_image', 'class']).size().unstack(fill_value=0)
        total_counts = predictions_df.groupby('plot_image').size()
        
        plot_results = []
        for plot_filename, counts in plot_counts.iterrows():
            plot_match = re.search(r'plot_(\d+)', str(plot_filename))
            if plot_match:
                plot_number = int(plot_match.group(1))
                matching_plot = plot_gdf[plot_gdf['Plot'] == plot_number]
                if not matching_plot.empty:
                    plot_row = matching_plot.iloc[0]
                    result_row = {'Plot': plot_number, 'Accession': plot_row.get('Accession', 'Unknown'), 'geometry': plot_row['geometry']}
                    result_row['total_detections'] = total_counts.get(plot_filename, 0)
                    for class_name, count in counts.items():
                        result_row[f'{class_name}_count'] = count
                    plot_results.append(result_row)
        
        if not plot_results: return None
        
        results_gdf = gpd.GeoDataFrame(plot_results, crs=plot_gdf.crs)
        traits_dir = os.path.join(data_root_dir, 'Processed', year, experiment, location, population, date, platform, sensor, agrowstitch_dir)
        os.makedirs(traits_dir, exist_ok=True)
        traits_geojson_path = os.path.join(traits_dir, 'traits.geojson')
        results_gdf.to_file(traits_geojson_path, driver='GeoJSON')
        return traits_geojson_path
    except Exception as e:
        print(f"ERROR in create_traits_geojson: {e}")
        traceback.print_exc()
        return None

def run_local_inference_worker(model_name, year, experiment, location, population, date, platform, sensor, agrowstitch_dir, data_root_dir, model_task='detection', confidence=0.25):
    """Worker function to run local inference in background."""
    global local_inference_status
    try:
        local_inference_status = {'running': True, 'progress': 0, 'message': 'Starting local inference...', 'error': None, 'results': None, 'completed': False}
        model_path = os.path.join(data_root_dir, 'models', model_name, 'model_weights.pt')
        if not os.path.exists(model_path): raise FileNotFoundError(f"Model not found: {model_path}")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, detected_model_type = load_local_model(model_path, device)
        
        if model_task == 'auto':
            model_task = detected_model_type
        
        local_inference_status.update({'message': f'Model loaded. Running {model_task} inference...', 'progress': 10})
        
        images_dir = os.path.join(data_root_dir, 'Processed', year, experiment, location, population, date, platform, sensor, agrowstitch_dir)
        if agrowstitch_dir == 'Plot_Images': # Adjust path for plot_images
             images_dir = os.path.join(data_root_dir, 'Intermediate', year, experiment, location, population, 'plot_images', date)

        if not os.path.exists(images_dir): raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        file_prefix = 'plot_' if agrowstitch_dir == 'Plot_Images' else 'full_res_mosaic_temp_plot_'
        image_files = [f for f in os.listdir(images_dir) if f.startswith(file_prefix) and f.endswith('.png')]
        if not image_files: raise FileNotFoundError(f"No plot images found in: {images_dir}")
        total_images_found = len(image_files)
        
        local_inference_status['message'] = f'Processing {len(image_files)} images...'
        all_predictions = []
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(images_dir, image_file)
            with Image.open(image_path) as img:
                w, h = img.size
            
            image_predictions = []
            if max(w, h) > 1280:
                crops = crop_image_with_overlap(image_path)
                temp_dirs = set()
                for crop_info in crops:
                    temp_dirs.add(crop_info['temp_dir'])
                    crop_preds = run_inference_on_image(model, crop_info['crop_path'], model_task, confidence)
                    image_predictions.extend(transform_predictions_to_plot_coordinates(crop_preds, crop_info))
                for temp_dir in temp_dirs:
                    shutil.rmtree(temp_dir)
                image_predictions = apply_nms(image_predictions)
            else:
                image_predictions = run_inference_on_image(model, image_path, model_task, confidence)
            
            for pred in image_predictions:
                pred['plot_image'] = image_file
                pred['model_name'] = model_name
                pred['model_task'] = model_task
                if 'points' in pred and pred['points']:
                    pred['points'] = json.dumps(pred['points'])
            
            all_predictions.extend(image_predictions)
            local_inference_status['progress'] = int(20 + (i + 1) / len(image_files) * 70)
            local_inference_status['message'] = f'Processed {i + 1}/{len(image_files)} images...'
        
        if not all_predictions: raise ValueError("No predictions generated")
        
        local_inference_status.update({'message': 'Saving results...', 'progress': 90})
        predictions_df = pd.DataFrame(all_predictions)
        
        if agrowstitch_dir == 'Plot_Images':
            results_dir = os.path.join(data_root_dir, 'Intermediate', year, experiment, location, population, 'plot_images', date)
            csv_filename = f"{date}_Plot_Images_local_{model_name}_{model_task}.csv"
        else:
            results_dir = os.path.join(data_root_dir, 'Processed', year, experiment, location, population, date, platform, sensor)
            csv_filename = f"{date}_{platform}_{sensor}_{agrowstitch_dir}_local_{model_name}_{model_task}.csv"
        
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(results_dir, csv_filename)
        predictions_df.to_csv(csv_path, index=False)
        
        traits_geojson_path = create_traits_geojson(predictions_df, data_root_dir, year, experiment, location, population, date, platform, sensor, agrowstitch_dir)
        
        class_counts = predictions_df['class'].value_counts().to_dict()
        results = {
            'csvPath': csv_path, 'geojsonPath': traits_geojson_path,
            'totalPlots': predictions_df['plot_image'].nunique(), 'totalPredictions': len(predictions_df),
            'labels': [{'name': name, 'count': count} for name, count in class_counts.items()],
            'hasSegmentation': model_task == 'segmentation', 'model_name': model_name, 'model_task': model_task
        }
        local_inference_status.update({'results': results, 'completed': True, 'running': False, 'progress': 100, 'message': 'Local inference completed!'})
    except Exception as e:
        print(f"ERROR in local inference: {e}")
        traceback.print_exc()
        local_inference_status.update({'error': str(e), 'running': False, 'completed': False, 'message': f'Error: {str(e)}'})

# Flask route functions
def run_local_inference_endpoint(file_app, data_root_dir):
    """Run local inference on stitched plot images."""
    @file_app.route('/run_local_inference', methods=['POST'])
    def run_local_inference():
        global local_inference_status
        if local_inference_status['running']:
            return jsonify({'error': 'Local inference is already running'}), 400
        try:
            data = request.json
            required = ['model_name', 'year', 'experiment', 'location', 'population', 'date', 'platform', 'sensor', 'agrowstitch_dir']
            if not all(k in data for k in required):
                return jsonify({'error': 'Missing required parameters'}), 400
            
            thread = threading.Thread(target=run_local_inference_worker, kwargs={**data, 'data_root_dir': data_root_dir})
            thread.start()
            return jsonify({'message': 'Local inference started', 'status': 'running'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

def get_local_inference_progress_endpoint(file_app):
    """Get current local inference progress."""
    @file_app.route('/get_local_inference_progress', methods=['GET'])
    def get_local_inference_progress():
        return jsonify(local_inference_status)

def get_local_models_endpoint(file_app, data_root_dir):
    """Get list of available local models."""
    @file_app.route('/get_local_models', methods=['GET'])
    def get_local_models():
        try:
            models_dir = os.path.join(data_root_dir, 'models')
            if not os.path.exists(models_dir): return jsonify({'models': []})
            
            models = []
            for model_name in os.listdir(models_dir):
                model_dir = os.path.join(models_dir, model_name)
                if os.path.isdir(model_dir):
                    model_weights_path = os.path.join(model_dir, 'model_weights.pt')
                    metadata_path = os.path.join(model_dir, 'model_metadata.json')
                    if os.path.exists(model_weights_path):
                        model_info = {'name': model_name}
                        if os.path.exists(metadata_path):
                            with open(metadata_path, 'r') as f:
                                model_info.update(json.load(f))
                        models.append(model_info)
            return jsonify({'models': models})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

def upload_local_model_endpoint(file_app, data_root_dir):
    """Upload a new local model and generate a persistent metadata file."""
    def _inspect_and_create_metadata(model_name, temp_file_path):
        """Helper function to inspect a model file and save its metadata."""
        models_dir = os.path.join(data_root_dir, 'models')
        model_dir = os.path.join(models_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        final_model_path = os.path.join(model_dir, 'model_weights.pt')
        shutil.move(temp_file_path, final_model_path)

        try:
            model, detected_model_type, classes = None, 'detection', []
            if ULTRALYTICS_AVAILABLE:
                try:
                    model = YOLO(final_model_path)
                    if hasattr(model, 'task'): detected_model_type = 'segmentation' if model.task == 'segment' else 'detection'
                    if hasattr(model, 'names'): classes = list(model.names.values())
                except Exception: model = None
            
            if model is None:
                model = torch.load(final_model_path, map_location='cpu')
                if hasattr(model, 'model') and hasattr(model.model, 'args') and hasattr(model.model.args, 'task'):
                    detected_model_type = 'segmentation' if model.model.args.task == 'segment' else 'detection'
                if hasattr(model, 'names'):
                    classes = list(model.names.values()) if isinstance(model.names, dict) else model.names

            model_info = {'name': model_name, 'path': final_model_path, 'size': os.path.getsize(final_model_path), 'task': detected_model_type, 'classes': classes}
            with open(os.path.join(model_dir, 'model_metadata.json'), 'w') as f:
                json.dump(model_info, f, indent=2)
            return {'message': f'Model "{model_name}" uploaded successfully', 'model_info': model_info}
        except Exception as e:
            if os.path.exists(final_model_path): os.remove(final_model_path)
            if os.path.exists(model_dir) and not os.listdir(model_dir): os.rmdir(model_dir)
            return {'error': f'Invalid or unsupported model file: {str(e)}'}

    @file_app.route('/upload_local_model', methods=['POST'])
    def upload_local_model():
        try:
            if not request.is_json: return jsonify({'error': 'Invalid request format'}), 400
            data = request.json
            model_name = data.get('model_name', '').strip()
            uploaded_file = data.get('uploaded_file', '')
            if not model_name or not uploaded_file: return jsonify({'error': 'Missing model name or filename'}), 400

            temp_upload_dir = os.path.join(data_root_dir, 'Raw', 'temp', 'model_uploads')
            temp_file_path = None
            if os.path.exists(temp_upload_dir):
                for timestamp_dir in os.listdir(temp_upload_dir):
                    potential_file = os.path.join(temp_upload_dir, timestamp_dir, uploaded_file)
                    if os.path.exists(potential_file):
                        temp_file_path = potential_file
                        break
            
            if not temp_file_path: return jsonify({'error': f'Uploaded file not found: {uploaded_file}'}), 400
            result = _inspect_and_create_metadata(model_name, temp_file_path)
            return jsonify(result), 400 if 'error' in result else 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500

def get_local_inference_results_endpoint(file_app, data_root_dir):
    """Get available local inference results for browsing and visualization."""
    @file_app.route('/get_local_inference_results', methods=['POST'])
    def get_local_inference_results():
        try:
            data = request.json
            required = ['year', 'experiment', 'location', 'population']
            if not all(k in data for k in required):
                return jsonify({'error': 'Missing required parameters'}), 400
            
            results = []
            dates_dir = os.path.join(data_root_dir, 'Processed', *[data[k] for k in required])
            if os.path.exists(dates_dir):
                for date in os.listdir(dates_dir):
                    # ... logic to find and parse local inference CSVs ...
                    pass # Placeholder for brevity
            
            # Placeholder for Plot_Images results in Intermediate
            
            results.sort(key=lambda x: x['timestamp'], reverse=True)
            return jsonify({'results': results})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

def get_local_plot_predictions_endpoint(file_app, data_root_dir):
    """Get predictions for a specific plot image to overlay bounding boxes (local models)."""
    @file_app.route('/get_local_plot_predictions', methods=['POST'])
    def get_local_plot_predictions():
        try:
            data = request.json
            # ... logic to find the right CSV based on data ...
            csv_path = None # Placeholder for path finding logic
            if not csv_path or not os.path.exists(csv_path):
                return jsonify({'error': 'Local inference results not found'}), 404
            
            df = pd.read_csv(csv_path)
            plot_predictions = df[df['plot_image'] == data.get('plot_filename')]
            
            predictions, class_counts = [], {}
            for _, row in plot_predictions.iterrows():
                pred = row.to_dict()
                if 'points' in pred and pd.notna(pred['points']):
                    try: pred['points'] = json.loads(pred['points'])
                    except: pred.pop('points')
                predictions.append(pred)
                class_counts[row['class']] = class_counts.get(row['class'], 0) + 1
            
            return jsonify({'predictions': predictions, 'class_counts': class_counts})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

def delete_local_inference_results_endpoint(file_app, data_root_dir):
    """Delete local inference results files for a specific model run."""
    @file_app.route('/delete_local_inference_results', methods=['POST'])
    def delete_local_inference_results():
        # Placeholder for delete logic
        return jsonify({'message': 'Deletion endpoint placeholder'})

def register_local_inference_routes(file_app, data_root_dir):
    """Register all local inference routes"""
    run_local_inference_endpoint(file_app, data_root_dir)
    get_local_inference_progress_endpoint(file_app)
    get_local_models_endpoint(file_app, data_root_dir)
    upload_local_model_endpoint(file_app, data_root_dir)
    get_local_inference_results_endpoint(file_app, data_root_dir)
    get_local_plot_predictions_endpoint(file_app, data_root_dir)
    delete_local_inference_results_endpoint(file_app, data_root_dir)