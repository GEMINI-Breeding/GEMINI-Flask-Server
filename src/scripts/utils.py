import asyncio
from concurrent.futures import ThreadPoolExecutor
import subprocess
import os
import shutil
import platform
import select
import yaml
from pathlib import Path
import pandas as pd
import random
import re
import string
from datetime import datetime, timezone

def _copy_image(src_folder, dest_folder, image_name):
    
    src_path = os.path.join(src_folder, image_name)
    dest_path = os.path.join(dest_folder, image_name)

    if not os.path.exists(dest_path):
        shutil.copy(src_path, dest_path)

def check_nvidia_smi():
    '''
    Check if nvidia-smi is installed on the system.
    Returns False automatically on macOS as it doesn't support NVIDIA GPUs.
    '''

    # Check operating system first
    if platform.system() == 'Darwin':  # 'Darwin' is the system name for macOS
        return False
        
    # For other systems, check using docker nvidia-smi
    try:
        output = subprocess.check_output(['docker', 'run', '--rm', '--gpus', 'all', 'nvidia/cuda:11.0.3-base', 'nvidia-smi'])
        if 'NVIDIA-SMI' in output.decode('utf-8'):
            return True
        else:
            return False
    except Exception as e:
        return False

def build_nested_structure_sync(path, current_depth=0, max_depth=2):
    if current_depth >= max_depth:
        return {}
    
    structure = {}
    for child in path.iterdir():
        if child.is_dir():
            structure[child.name] = build_nested_structure_sync(child, current_depth+1, max_depth)
    return structure

async def build_nested_structure(path, current_depth=0, max_depth=2):
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, build_nested_structure_sync, path, current_depth, max_depth)

async def process_directories_in_parallel(base_dir, max_depth=2):
    directories = [d for d in base_dir.iterdir() if d.is_dir()]
    tasks = [build_nested_structure(d, 0, max_depth) for d in directories]
    nested_structures = await asyncio.gather(*tasks)
    
    combined_structure = {}
    for d, structure in zip(directories, nested_structures):
        combined_structure[d.name] = structure
    
    return combined_structure

def dms_to_decimal(dms_str):
    parts = dms_str.split()
    degrees = float(parts[0])
    minutes = float(parts[2].replace('\'', ''))
    seconds = float(parts[3].replace('\"', ''))
    direction = parts[-1]
    
    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if direction in ['S', 'W']:
        decimal = -decimal
    return decimal

def stream_output(process):
    """Function to read the process output and errors in real-time."""
    while True:
        reads = [process.stdout.fileno(), process.stderr.fileno()]
        ret = select.select(reads, [], [])

        for fd in ret[0]:
            if fd == process.stdout.fileno():
                output = process.stdout.readline()
                if output:
                    print("Output:", output.decode('utf-8').strip())
            if fd == process.stderr.fileno():
                error_output = process.stderr.readline()
                if error_output:
                    print("Error:", error_output.decode('utf-8').strip())

        if process.poll() is not None:
            break  # Break loop if process ends

    # Close stdout and stderr after reading
    process.stdout.close()
    process.stderr.close()

def build_nested_structure_sync_from_db(dir_index, path, current_depth=0, max_depth=2):
    """Build nested structure using DirectoryIndex database"""
    
    if current_depth >= max_depth:
        return {}
    
    structure = {}
    
    # Get children directories from database
    children = dir_index.get_children(str(path), directories_only=True)
    
    for child_name in children:
        child_path = path / child_name
        structure[child_name] = build_nested_structure_sync_from_db(dir_index, child_path, current_depth + 1, max_depth)
    
    return structure

async def build_nested_structure_from_db(dir_index, path, current_depth=0, max_depth=2):
    """Async wrapper for database-based nested structure building"""
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, build_nested_structure_sync_from_db, dir_index, path, current_depth, max_depth)

async def process_directories_in_parallel_from_db(dir_index, base_dir, max_depth=2):
    """Process directories using DirectoryIndex database"""

    # Get top-level directories from database
    top_level_dirs = dir_index.get_children(str(base_dir), directories_only=True)
    
    # Convert to Path objects
    directories = [base_dir / dir_name for dir_name in top_level_dirs]
    
    # Build nested structures for each top-level directory
    tasks = [build_nested_structure_from_db(dir_index, d, 0, max_depth) for d in directories]
    nested_structures = await asyncio.gather(*tasks)
    
    combined_structure = {}
    for d, structure in zip(directories, nested_structures):
        combined_structure[d.name] = structure
    
    return combined_structure

### ROVER EXTRACT PLANTS ###
def update_or_add_entry(data, key, new_values):
    if key in data:
        # Update existing entry
        data[key].update(new_values)
    else:
        # Add new entry
        data[key] = new_values

def split_data(labels, images, test_size=0.2):
    # Calculate split index
    split_index = int(len(labels) * (1 - test_size))
    
    # Split the labels and images into train and validation sets
    labels_train = labels[:split_index]
    labels_val = labels[split_index:]
    
    images_train = images[:split_index]
    images_val = images[split_index:]
    
    return labels_train, labels_val, images_train, images_val

def copy_files_to_folder(source_files, target_folder):
    for source_file in source_files:
        target_file = target_folder / source_file.name
        if not target_file.exists():
            shutil.copy(source_file, target_file)
            
def remove_files_from_folder(folder):
    for file in folder.iterdir():
        if file.is_file():
            file.unlink()

def prepare_labels(annotations, images_path):
    try:
        # global data_root_dir, labels_train_folder, labels_val_folder, images_train_folder, images_val_folder
        
        # path to labels
        labels_train_folder = annotations.parent/'labels'/'train'
        labels_val_folder = annotations.parent/'labels'/'val'
        images_train_folder = annotations.parent/'images'/'train'
        images_val_folder = annotations.parent/'images'/'val'
        labels_train_folder.mkdir(parents=True, exist_ok=True)
        labels_val_folder.mkdir(parents=True, exist_ok=True)
        images_train_folder.mkdir(parents=True, exist_ok=True)
        images_val_folder.mkdir(parents=True, exist_ok=True)

        # obtain path to images
        images = list(images_path.rglob('*.jpg')) + list(images_path.rglob('*.png'))
        
        # split images to train and val
        labels = list(annotations.glob('*.txt'))
        label_stems = set(Path(label).stem for label in labels)
        filtered_images = [image for image in images if Path(image).stem in label_stems]
        labels_train, labels_val, images_train, images_val = split_data(labels, filtered_images)

        # link images and labels to folder
        copy_files_to_folder(labels_train, labels_train_folder)
        copy_files_to_folder(labels_val, labels_val_folder)
        copy_files_to_folder(images_train, images_train_folder)
        copy_files_to_folder(images_val, images_val_folder)
        
        # check if images_train_folder and images_val_folder are not empty
        if not any(images_train_folder.iterdir()) or not any(images_val_folder.iterdir()):
            return False
        else:
            return True
        
    except Exception as e:
        print(f'Error preparing labels for training: {e}')

### ROVER MODEL TRAINING ###
def check_model_details(key, value = None):
    # get base folder, args file and results file
    base_path = key.parent.parent
    args_file = base_path / 'args.yaml'
    results_file = base_path / 'results.csv'
    
    # get epochs, batch size and image size
    values = []
    with open(args_file, 'r') as file:
        args = yaml.safe_load(file)
        epochs = args.get('epochs')
        batch = args.get('batch')
        imgsz = args.get('imgsz')
        
        values.extend([epochs, batch, imgsz])
    
    # get mAP of model
    df = pd.read_csv(results_file, delimiter=',\s+', engine='python')
    mAP = round(df['metrics/mAP50(B)'].iloc[-1], 2)  # Get the last value in the column
    values.extend([mAP])
    
    # get run name
    run = base_path.name
    match = re.search(r'-([A-Za-z0-9]+)$', run)
    id = match.group(1)
    
    # get date(s)
    if value is not None:
        date = ', '.join(value)
    else:
        date = None
    
    # get platform
    platform = base_path.parts[-3]
    
    # get sensor
    sensor = base_path.parts[-2].split()[0]
    
    # collate details
    details = {'id': id, 'dates': date, 'platform': platform, 'sensor': sensor, 'epochs': epochs, 'batch': batch, 'imgsz': imgsz, 'map': mAP}
    
    return details

def generate_hash(trait, length=6):
    """Generate a hash for model where it starts with the trait followed by a random string of characters.

    Args:
        trait (str): trait to be analyzed (plant, flower, pod, etc.)
        length (int, optional): Length for random sequence. Defaults to 5.
    """
    random_sequence = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    hash_id = f"{trait}-{random_sequence}"
    return hash_id

def normalize_path(path):
    if not path:
        return path
    normalized = os.path.abspath(path)
    if len(normalized) > 1 and normalized.endswith(os.sep):
        normalized = normalized.rstrip(os.sep)
    return normalized


# Convert timestamps to Unix timestamps for KD tree matching
def convert_to_unix_timestamp(timestamp_str):
    try:
        # Try different timestamp formats with optional timezone
        formats = [
            '%Y:%m:%d %H:%M:%S.%f %z',  # EXIF format with microseconds and timezone
            '%Y:%m:%d %H:%M:%S %z',      # EXIF format with timezone
            '%Y:%m:%d %H:%M:%S.%f',      # EXIF format with microseconds
            '%Y:%m:%d %H:%M:%S',         # EXIF format
            '%Y-%m-%d %H:%M:%S %z',      # Standard format with timezone
            '%Y-%m-%d %H:%M:%S',         # Standard format
            '%Y/%m/%d %H:%M:%S %z',      # Alternative format with timezone
            '%Y/%m/%d %H:%M:%S',         # Alternative format
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(str(timestamp_str), fmt)
                # If no timezone info, assume UTC
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.timestamp()
            except ValueError:
                continue
            except Exception as e:
                print(f"Error parsing timestamp: {e}")
        
        # If no format works, try parsing as float (already Unix timestamp)
        return float(timestamp_str)
    except Exception as e:
        print(f"Failed to convert timestamp: {e}")
        return None