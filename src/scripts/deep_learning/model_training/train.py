import argparse
import yaml
import sys
import os
import random
import string
import torch

from ultralytics import YOLO
from pathlib import Path
from typing import List

def edit_data(
    data: Path,
    new_path: dict
) -> None:
    """
    Edits existing data.yaml file with new path and labels.
    
    Args:
        data (Path): Path to data.yaml file.
        new_path (dict): dictionary with new values to update.
    """
    
    try:
        with open(str(data), 'r') as file:
            d = yaml.safe_load(file)
            
        # update path value in yaml file
        for key, value in new_path.items():
            keys = key.split('.')
            temp_d = d
            for k in keys[:-1]:
                if k not in temp_d:
                    raise KeyError(f"Key '{key}' not found in the YAML file.")
                temp_d = temp_d[k]
            temp_d[keys[-1]] = value
            
        print(f"YAML file '{data}' has been updated.")
            
        # save new yaml file
        with open(data, 'w') as file:
            yaml.dump(d, file, default_flow_style=False)
        
    except Exception as e:
        print(f'An error occured: {e}')

def load_yaml_config(
    config_file: Path
) -> dict:
    """
    Loads training configuration file into dictionary
    
    Args:
        config_file (Path): Path to config.yaml file.
    """
    
    # check if config file exists
    if not config_file.exists():
        print(f'Config file {config_file} not found.')
        sys.exit(1)
    
    # opens config.yaml file and returns a dictionary
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def generate_hash(trait, length=6):
    """Generate a hash for model where it starts with the trait followed by a random string of characters.

    Args:
        trait (str): trait to be analyzed (plant, flower, pod, etc.)
        length (int, optional): Length for random sequence. Defaults to 5.
    """
    random_sequence = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    hash_id = f"{trait}-{random_sequence}"
    return hash_id
    

def main(
    pretrained: Path,
    images: Path,
    config_path: Path,
    data: Path,
    save: Path,
    labels: List[str],
    image_size: int,
    epochs: int,
    batch_size: int,
    sensor: str,
    dates: str,
    trait: str
) -> None:
    
        # edit data.yaml file with update path to images
        new_path: dict = {
            'path': str(images),
            'names': labels,
            'nc': len(labels)
        }
        edit_data(data, new_path)
        
        # initialize project and update config
        project = f'{sensor} {trait} Detection'
        config = load_yaml_config(config_file=config_path)
        version = generate_hash(trait)
        while (save / project / Path(f'{version}')).exists():
            version = generate_hash(trait)
        name = f'{version}'
        new_values: dict = {
            'batch': batch_size,
            'epochs': epochs,
            'imgsz': image_size,
            'project': f'{save}/{project}',
            'name': name
        }
        config.update(new_values)
        with open(config_path, 'w') as file:
            yaml.dump(config, file)
        
        # load pretrained model
        model = YOLO(pretrained)

        # train the model
        model.train(data=data, 
                    batch=config['batch'],
                    epochs=config['epochs'], 
                    imgsz=config['imgsz'], 
                    device=config['device'],
                    project=config['project'],
                    name=config['name'],
                    patience=config['patience'],
                    workers=config['workers'],
                    close_mosaic=0,
                    verbose=True)

        # export the model
        model.export()
        
        # log any data needed for future reference
        log = {'dates': dates}
        log_save = save / project / Path(f'{version}') / 'logs.yaml'
        with open(log_save, 'w') as file:
            yaml.dump(log, file, default_flow_style=False)

if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument('--pretrained', type=Path, required=True,
                    help='Path to pretrained YOLOv8 weights')
    ap.add_argument('--images', type=Path, required=True,
                    help='Path to training and validation dataset') 
    ap.add_argument('--save', type=Path, required=True,
                    help='Path to save weights and results.')
    ap.add_argument('--sensor', type=str, required=True,
                    help='Type of sensor used.')
    ap.add_argument('--dates', nargs='+', required=True,
                    help='List of dates used for training.')
    ap.add_argument('--trait', type=str, required=True,
                    help='Name of trait the user is annotating.')
    ap.add_argument('--image-size', type=int, default=640,
                    help='Image size for training.')
    ap.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train the model.')
    ap.add_argument('--batch-size', type=int, default=32,
                    help='Batch size per epoch.')
    ap.add_argument('--labels', nargs='+', default=['0','1'],
                    help='Labels used for training')
    args = ap.parse_args()
    
    # set path to directory of script
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    print(f'CWD: {dname}')
    # os.chdir(dname)
    
    # defaults
    config = Path(f'{dname}/config.yaml')
    data = Path(f'{dname}/data.yaml')
    pretrained = Path(f'{dname}/yolov8n.pt')
    print(f'Config file: {config}')
    print(f'Data file: {data}')
    print(f'Pretrained model: {pretrained}')
    
    main(args.pretrained, args.images, config, data, \
            args.save, args.labels, args.image_size, args.epochs, \
                args.batch_size, args.sensor, args.dates, args.trait)