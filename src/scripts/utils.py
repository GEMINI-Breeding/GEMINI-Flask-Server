import asyncio
from concurrent.futures import ThreadPoolExecutor
import subprocess
import os
import shutil

def _copy_image(src_folder, dest_folder, image_name):
    
    src_path = os.path.join(src_folder, image_name)
    dest_path = os.path.join(dest_folder, image_name)

    if not os.path.exists(dest_path):
        shutil.copy(src_path, dest_path)

def check_nvidia_smi():
    '''
    Check if nvidia-smi is installed on the system.
    '''
    # Check the output of "docker run --rm --gpus all nvidia/cuda:11.0.3-base nvidia-smi"
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