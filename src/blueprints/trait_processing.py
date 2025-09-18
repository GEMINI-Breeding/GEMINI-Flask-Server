# Standard library imports
import os
import re
import subprocess
import threading
import time
import yaml
import requests
from pathlib import Path
import re

# Third-party library imports

import pandas as pd
import numpy as np
from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename


from scripts.utils import stream_output
from scripts.utils import update_or_add_entry, prepare_labels, remove_files_from_folder, check_model_details
from scripts.utils import generate_hash
        
# Paths to scripts
TRAIN_MODEL = os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts/deep_learning/model_training/train.py'))
LOCATE_PLANTS = os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts/deep_learning/trait_extraction/locate.py'))
EXTRACT_TRAITS = os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts/deep_learning/trait_extraction/extract.py'))

model_bp = Blueprint('model', __name__)

### CVAT #### 
@model_bp.route('/start_cvat', methods=['POST'])
def start_cvat():
    # global data_root_dir
    clone_dir = os.path.join(data_root_dir, 'cvat')
    
    # Create the directory if it doesn't exist
    os.makedirs(clone_dir, exist_ok=True)
    
    # Define the path for the compose directory
    compose_dir = os.path.join(clone_dir, 'cvat')

    try:
        # Get a list of all containers with 'cvat' or 'traefik' in their name
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=cvat", "--format", "{{.Names}}"],
            stdout=subprocess.PIPE
        )
        cvat_containers = result.stdout.decode('utf-8').strip().split('\n')
        cvat_containers = [container for container in cvat_containers if container]  # filter out empty strings

        # Add traefik to the list of containers to restart
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=traefik", "--format", "{{.Names}}"],
            stdout=subprocess.PIPE
        )
        traefik_containers = result.stdout.decode('utf-8').strip().split('\n')
        traefik_containers = [container for container in traefik_containers if container]  # filter out empty strings

        # Combine the lists of containers
        all_containers = cvat_containers + traefik_containers

    except Exception as e:
        return jsonify({"error": f"Error checking for CVAT or Traefik containers: {str(e)}"}), 404

    try:
        if all_containers:
            # Restart each container with 'cvat' or 'traefik' in its name
            print(f"Restarting containers: {all_containers}")
            for container in all_containers:
                subprocess.run(["docker", "restart", container])
            print("All relevant containers have been restarted.")
        else:
            # Clone the repository if needed and run docker-compose
            if not os.path.exists(compose_dir):
                subprocess.run(
                    ["git", "clone", "https://github.com/cvat-ai/cvat"], cwd=clone_dir
                )
                
            # Check if docker-compose.yml exists before starting docker-compose
            compose_file = os.path.join(compose_dir, 'docker-compose.yml')
            if not os.path.exists(compose_file):
                return jsonify({"error": "docker-compose.yml not found in the cloned repository"}), 404
            
            # Start CVAT with docker-compose
            subprocess.run(
                ["docker-compose", "up", "-d"], cwd=compose_dir
            )
            print("Starting CVAT container with docker-compose...")

        # Wait for specific services to be fully up and running
        services_to_check = ["cvat_server", "cvat_ui"]
        max_retries = 30
        for i in range(max_retries):
            # Check the status of the containers
            result = subprocess.run(
                ["docker-compose", "ps", "--services", "--filter", "status=running"], cwd=compose_dir, stdout=subprocess.PIPE
            )
            running_services = result.stdout.decode('utf-8').strip().split('\n')

            # Check if all expected services are running
            if all(service in running_services for service in services_to_check):
                print("All required CVAT services are running.")
                break

            print(f"Waiting for CVAT services to start... ({i + 1}/{max_retries})")
            time.sleep(5)  # Wait 5 seconds before checking again

        if i == max_retries - 1:
            return jsonify({"error": "CVAT services failed to start in time"}), 500

        # Create superuser via docker exec
        subprocess.run(
            ["docker", "exec", "-it", "cvat_server", "bash", "-c", "'python3 ~/manage.py createsuperuser'"]
        )
        print("CVAT superuser created.")

        # Poll the CVAT server to check if it's running
        cvat_url = "http://localhost:8080/api/server/about"
        for i in range(max_retries):
            try:
                response = requests.get(cvat_url)
                if response.status_code == 200:
                    print("CVAT server is up and running.")
                    break
            except requests.exceptions.RequestException:
                pass  # Server is not ready yet

            print(f"Waiting for CVAT to start... ({i + 1}/{max_retries})")
            time.sleep(5)  # Wait 5 seconds before checking again

        # If the server didn't start within the max retries, return an error
        if i == max_retries - 1:
            return jsonify({"error": "CVAT server failed to start in time"}), 500

        return jsonify({"status": "CVAT and Traefik containers restarted and superuser created"})
    except Exception as e:
        print(f"Error starting CVAT or Traefik containers: {str(e)}")
        return jsonify({"error": f"Error starting CVAT or Traefik containers: {str(e)}"}), 404

### ROVER LABELS PREPARATION ###
@model_bp.route('/check_labels/<path:dir_path>', methods=['GET'])
def check_labels(dir_path):
    # global data_root_dir
    data = []
    
    # get labels path
    labels_path = Path(data_root_dir)/dir_path

    if labels_path.exists() and labels_path.is_dir():
        # Use glob to find all .txt files in the directory
        txt_files = list(labels_path.glob('*.txt'))
        
        # Check if there are more than one .txt files
        if len(txt_files) > 1:
            data.append(str(labels_path))

    return jsonify(data)

@model_bp.route('/check_existing_labels', methods=['POST'])
def check_existing_labels():
    # global data_root_dir
    
    data = request.json
    fileList = data['fileList']
    dirPath = data['dirPath']
    full_dir_path = os.path.join(data_root_dir, dirPath)

    # existing_files = set(os.listdir(full_dir_path)) if os.path.exists(full_dir_path) else set()
    existing_files = [file.name for file in Path(full_dir_path).rglob('*.txt')]
    new_files = [file for file in fileList if file not in existing_files]

    print(f"Uploading {str(len(new_files))} out of {str(len(fileList))} files to {dirPath}")

    return jsonify(new_files), 200

@model_bp.route('/upload_trait_labels', methods=['POST'])
def upload_trait_labels():
    dir_path = request.form.get('dirPath')
    full_dir_path = os.path.join(data_root_dir, dir_path)
    os.makedirs(full_dir_path, exist_ok=True)

    uploaded_files = []
    for file in request.files.getlist("files"):
        filename = secure_filename(file.filename)
        file_path = os.path.join(full_dir_path, filename)
        print(f'Saving {file_path}...')
        file.save(file_path)
        uploaded_files.append(file_path)

    # Update directory database after upload completion
    if uploaded_files and dir_db is not None:
        try:
            dir_db.force_refresh(full_dir_path)
            print(f"Updated directory database for: {full_dir_path}")
        except Exception as e:
            print(f"Error updating directory database: {e}")

    return jsonify({'message': 'Files uploaded successfully'}), 200





@model_bp.route('/get_model_info', methods=['POST'])
def get_model_info():
    data = request.json
    details_data = []
    
    # iterate through each existing model
    for key in data:
        details = check_model_details(Path(key), value = data[key])
        details_data.append(details)

    return jsonify(details_data)

def get_labels(labels_path):
    unique_labels = set()

    # Iterate over the files in the directory
    for filename in os.listdir(labels_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(labels_path, filename)
            with open(file_path, 'r') as file:
                for line in file:
                    label = line.split()[0]  # Extracting the label
                    unique_labels.add(label)

    sorted_unique_labels = sorted(unique_labels, key=lambda x: int(x))
    return list(sorted_unique_labels)

def scan_for_new_folders(save_path):
    global latest_data, training_stopped_event, new_folder, results_file
    known_folders = {os.path.join(save_path, f) for f in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, f))}

    while not training_stopped_event.is_set():  # Continue while training is ongoing
        # Check for new folders
        for folder_name in os.listdir(save_path):
            folder_path = os.path.join(save_path, folder_name)
            if os.path.isdir(folder_path) and folder_path not in known_folders:
                known_folders.add(folder_path)  # Add new folder to the set
                new_folder = folder_path  # Update global variable
                results_file = os.path.join(folder_path, 'results.csv')

                # Continuously check results.csv for updates
                while not os.path.isfile(results_file):
                    time.sleep(5)  # Check every 5 seconds

                # Periodically read results.csv for updates
                while os.path.exists(results_file) and os.path.isfile(results_file):
                    try:
                        df = pd.read_csv(results_file, delimiter=',\s+', engine='python')
                        latest_data['epoch'] = int(df['epoch'].iloc[-1])  # Update latest epoch
                        latest_data['map'] = df['metrics/mAP50(B)'].iloc[-1]  # Update latest mAP
                    except Exception as e:
                        print(f"Error reading {results_file}: {e}")
                    time.sleep(5)  # Update every 30 seconds

        time.sleep(5)  # Check for new folders every 10 seconds
        
@model_bp.route('/get_progress', methods=['GET'])
def get_training_progress():
    for key, value in latest_data.items():
        if isinstance(value, np.int64):
            latest_data[key] = int(value)
    print(latest_data)
    return jsonify(latest_data)

@model_bp.route('/train_model', methods=['POST'])
def train_model():
    # global data_root_dir, latest_data, training_stopped_event, new_folder, train_labels, training_process
    
    try:
        # receive the parameters
        epochs = int(request.json['epochs'])
        batch_size = int(request.json['batchSize'])
        image_size = int(request.json['imageSize'])
        location = request.json['location']
        population = request.json['population']
        date = request.json['date']
        trait = request.json['trait']
        sensor = request.json['sensor']
        platform = request.json['platform']
        year = request.json['year']
        experiment = request.json['experiment']
        
        # prepare labels
        annotations = Path(data_root_dir)/'Intermediate'/year/experiment/location/population/date/platform/sensor/f'Labels/{trait} Detection/annotations'
        all_images = Path(data_root_dir)/'Raw'/year/experiment/location/population/date/platform/sensor/'Images'
        # all_images = Path('/home/gemini/mnt/d/Annotations/Plant Detection/obj_train_data')
        check_if_images_exist = prepare_labels(annotations, all_images)
        # wait for 1 minute
        time.sleep(60)
        if check_if_images_exist == False:
            return jsonify({"error": "No images found for training. Press stop and upload images."}), 404
        
        # extract labels
        labels_path = Path(data_root_dir)/'Intermediate'/year/experiment/location/population/date/platform/sensor/f'Labels/{trait} Detection/labels/train'
        labels = get_labels(labels_path)
        labels_arg = " ".join(labels).split()
        
        # other training args
        pretrained = "yolov8n.pt"
        save_train_model = Path(data_root_dir)/'Intermediate'/year/experiment/location/population/'Training'/platform
        scan_save = Path(data_root_dir)/'Intermediate'/year/experiment/location/population/'Training'/platform/f'{sensor} {trait} Detection'
        scan_save = Path(scan_save)
        scan_save.mkdir(parents=True, exist_ok=True)
        latest_data['epoch'] = 0
        latest_data['map'] = 0
        training_stopped_event.clear()
        threading.Thread(target=scan_for_new_folders, args=(scan_save,), daemon=True).start()
        images = Path(data_root_dir)/'Intermediate'/year/experiment/location/population/date/platform/sensor/f'Labels/{trait} Detection'
        
        cmd = (
            f"python {TRAIN_MODEL} "
            f"--pretrained '{pretrained}' "
            f"--images '{images}' "
            f"--save '{save_train_model}' "
            f"--sensor '{sensor}' "
            f"--date '{date}' "
            f"--trait '{trait}' "
            f"--image-size '{image_size}' "
            f"--epochs '{epochs}' "
            f"--batch-size {batch_size} "
            f"--labels {' '.join(labels_arg)} "
        )
        print(cmd)
        
        training_process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        threading.Thread(target=stream_output, args=(training_process,), daemon=True).start()
        time.sleep(5)  # Wait for 5 seconds
        if training_process.poll() is None:
            print("Process started successfully and is running.")
        else:
            print("Process failed to start or exited immediately.")
        
        return jsonify({"message": "Training started"}), 202

    except subprocess.CalledProcessError as e:
        error_output = e.stderr.decode('utf-8')
        return jsonify({"error": error_output}), 500
    
@model_bp.route('/stop_training', methods=['POST'])
def stop_training():
    global training_stopped_event, new_folder, labels_train_folder, labels_val_folder, images_train_folder, images_val_folder, training_process

    try:        
        # stop training
        print('Training stopped by user.')
        if training_process is not None:
            training_process.terminate()
            training_process.wait()  # Optionally wait for the process to terminate
            print("Training process terminated.")
            training_process = None
        else:
            print("No training process running.")
            
        training_stopped_event.set()
        subprocess.run(f"rm -rf '{new_folder}'", check=True, shell=True)
        
        # unlink files
        remove_files_from_folder(labels_train_folder)
        remove_files_from_folder(labels_val_folder)
        remove_files_from_folder(images_train_folder)
        remove_files_from_folder(images_val_folder)
        
        return jsonify({"message": "Python process in container successfully stopped"}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": e.stderr.decode("utf-8")}), 500
    
@model_bp.route('/done_training', methods=['POST'])
def done_training():
    global training_stopped_event, new_folder, results_file, labels_train_folder, labels_val_folder, images_train_folder, images_val_folder, training_process
    # container_name = 'train'
    try:
        # stop training
        print('Training stopped by user.')
        # kill_cmd = f"docker exec {container_name} pkill -9 -f python"
        # subprocess.run(kill_cmd, shell=True)
        # print(f"Sent SIGKILL to Python process in {container_name} container.")
        if training_process is not None:
            training_process.terminate()
            training_process.wait()  # Optionally wait for the process to terminate
            print("Training process terminated.")
            training_process = None
        else:
            print("No training process running.")
            
        # subprocess.run(f"rm -rf '{new_folder}'", check=True, shell=True)
        # print(f"Removed {new_folder}")
        
        # unlink files
        remove_files_from_folder(labels_train_folder)
        remove_files_from_folder(labels_val_folder)
        remove_files_from_folder(images_train_folder)
        remove_files_from_folder(images_val_folder)
        training_stopped_event.set()
        results_file = ''
        return jsonify({"message": "Python process in container successfully stopped"}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": e.stderr.decode("utf-8")}), 500

### ROVER LOCATE PLANTS ###
def check_locate_details(key):
    
    # get base folder, args file and results file
    base_path = key.parent
    results_file = base_path / 'locate.csv'
    
    # get run name
    run = base_path.name
    match = re.search(r'-([A-Za-z0-9]+)$', run)
    id = match.group(1)
    
    # get model id
    with open(base_path/'logs.yaml', 'r') as file:
        data = yaml.safe_load(file)
    model_id = data['model']
    
    
    # get stand count
    df = pd.read_csv(results_file)
    stand_count = len(df)
    
    # get date
    date = data['date']
    
    # get platform
    platform = base_path.parts[-4]
    
    # get sensor
    sensor = base_path.parts[-3]
    
    # get mAP of model
    date_index = base_path.parts.index(date[0]) if date[0] in base_path.parts else None
    if date_index and date_index > 0:
        # Construct a new path from the parts up to the folder before the known date
        root_path = Path(*base_path.parts[:date_index])
        results_file = root_path / 'Training' / platform / f'{sensor} Plant Detection' / f'Plant-{model_id[0]}' / 'results.csv'
    df = pd.read_csv(results_file, delimiter=',\s+', engine='python')
    mAP = round(df['metrics/mAP50(B)'].iloc[-1], 2)
    # values.extend([mAP])
    
    # collate details
    details = {'id': id, 'model': model_id, 'count': stand_count, 'date': date, 'platform': platform, 'sensor': sensor, 'performance': mAP}
    
    return details

@model_bp.route('/get_locate_info', methods=['POST'])
def get_locate_info():
    data = request.json
    details_data = []
    
    # iterate through each existing model
    for key in data:
        details = check_locate_details(Path(key))
        details_data.append(details)

    return jsonify(details_data)

@model_bp.route('/get_locate_progress', methods=['GET'])
def get_locate_progress():
    global save_locate
    txt_file = save_locate/'locate_progress.txt'
    
    # Check if the file exists
    if os.path.exists(txt_file):
        with open(txt_file, 'r') as file:
            number = file.read().strip()
            latest_data['locate'] = int(number)
        return jsonify(latest_data)
    else:
        return jsonify({'error': 'Locate progress not found'}), 404



@model_bp.route('/locate_plants', methods=['POST'])
def locate_plants():
    # global data_root_dir, save_locate, locate_process
    
    # recieve parameters
    batch_size = int(request.json['batchSize'])
    location = request.json['location']
    population = request.json['population']
    date = request.json['date']
    platform = request.json['platform']
    sensor = request.json['sensor']
    year = request.json['year']
    experiment = request.json['experiment']
    model = request.json['model']
    id = request.json['id']
    
    # other args
    # container_dir = Path('/app/mnt/GEMINI-App-Data')
    images = Path(data_root_dir)/'Raw'/year/experiment/location/population/date/platform/sensor/'Images'
    disparity = Path(data_root_dir)/'Raw'/year/experiment/location/population/date/platform/sensor/'Disparity'
    configs = Path(data_root_dir)/'Raw'/year/experiment/location/population/date/platform/sensor/'Metadata'
    plotmap = Path(data_root_dir)/'Intermediate'/year/experiment/location/population/'Plot-Boundary-WGS84.geojson'
    
    # generate save folder
    version = generate_hash(trait='Locate')
    save_base = Path(data_root_dir)/'Intermediate'/year/experiment/location/population/date/platform/sensor/f'Locate'
    while (save_base / f'{version}').exists():
        version = generate_hash(trait='Locate')
    save_locate = Path(data_root_dir)/'Intermediate'/year/experiment/location/population/date/platform/sensor/f'Locate'/f'{version}'
    save_locate.mkdir(parents=True, exist_ok=True)
    model_path = Path(data_root_dir)/'Intermediate'/year/experiment/location/population/'Training'/f'{platform}'/'RGB Plant Detection'/f'Plant-{id}'/'weights'/'last.pt' # TODO: DEBUG
    # model_path = "/mnt/d/GEMINI-App-Data/Intermediate/2022/GEMINI/Davis/Legumes/Training/Amiga-Onboard/RGB Plant Detection/Plant-btRN26/weights/last.pt"
    
    # save logs file
    data = {"model": [id], "date": [date]}
    with open(save_locate/"logs.yaml", "w") as file:
        yaml.dump(data, file, default_flow_style=False)
        
    # create progress file
    with open(save_locate/"locate_progress.txt", "w") as file:
        pass
    
    # run locate
    cmd = (
        f"python -W ignore {LOCATE_PLANTS} "
        f"--images '{images}' --metadata '{configs}' --plotmap '{plotmap}' "
        f"--batch-size '{batch_size}' --model '{model_path}' --save '{save_locate}'"
    )

    if disparity.exists():
        cmd += " --skip-stereo"

    try:
        locate_process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        threading.Thread(target=stream_output, args=(locate_process,), daemon=True).start()
        time.sleep(5)  # Wait for 5 seconds
        if locate_process.poll() is None:
            print("Locate process started successfully and is running.")
            return jsonify({"message": "Locate started"}), 202
        else:
            print("Locate process failed to start or exited immediately.")
            return jsonify({"error": "Failed to start locate process." }), 404
    
    except subprocess.CalledProcessError as e:
        
        error_output = e.stderr.decode('utf-8')
        return jsonify({"error": error_output}), 404
    
@model_bp.route('/stop_locate', methods=['POST'])
def stop_locate():
    global save_locate, locate_process
    
    try:
        print('Locate stopped by user.')
        if locate_process is not None:
            locate_process.terminate()
            locate_process.wait()
            print("Locate process terminated.")
            locate_process = None
        else:
            print("No locate process running.")

        subprocess.run(f"rm -rf '{save_locate}'", check=True, shell=True)
        return jsonify({"message": "Python process in container successfully stopped"}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": e.stderr.decode("utf-8")}), 500
        
@model_bp.route('/get_extract_progress', methods=['GET'])
def get_extract_progress():
    global save_extract
    txt_file = save_extract/'extract_progress.txt'
    
    # Check if the file exists
    if os.path.exists(txt_file):
        with open(txt_file, 'r') as file:
            number = file.read().strip()
            latest_data['extract'] = int(number)
        return jsonify(latest_data)
    else:
        return jsonify({'error': 'Locate progress not found'}), 404
    
@model_bp.route('/extract_traits', methods=['POST'])
def extract_traits():
    # global data_root_dir, save_extract, temp_extract, model_id, summary_date, locate_id, trait_extract, extract_process
    
    try:
        # recieve parameters
        summary = request.json['summary']
        batch_size = int(request.json['batchSize'])
        model = request.json['model']
        trait = request.json['trait']
        trait_extract = request.json['trait']
        date = request.json['date']
        year = request.json['year']
        experiment = request.json['experiment']
        location = request.json['location']
        population = request.json['population']
        platform = request.json['platform']
        sensor = request.json['sensor']
        
        # extract model and summary information
        pattern = r"/[^/]+-([\w]+?)/weights"
        date_pattern = r"\b\d{4}-\d{2}-\d{2}\b"
        locate_pattern = r"Locate-(\w+)/"
        match = re.search(pattern, str(model))
        match_date = re.search(date_pattern, str(summary))
        match_locate_id = re.search(locate_pattern, str(summary))
        model_id = match.group(1)
        summary_date = match_date.group()
        locate_id = match_locate_id.group(1)
        
        # other args
        summary_path = Path(data_root_dir)/'Intermediate'/year/experiment/location/population/summary_date/platform/sensor/'Locate'/f'Locate-{locate_id}'/'locate.csv'
        model_path = Path(data_root_dir)/'Intermediate'/year/experiment/location/population/'Training'/f'{platform}'/'RGB Plant Detection'/f'Plant-{id}'/'weights'/'last.pt' # TODO: DEBUG
        images = Path(data_root_dir)/'Raw'/year/experiment/location/population/date/platform/sensor/'Images'
        disparity = Path(data_root_dir)/'Raw'/year/experiment/location/population/date/platform/sensor/'Disparity'
        plotmap = Path(data_root_dir)/'Intermediate'/year/experiment/location/population/'Plot-Boundary-WGS84.geojson'
        metadata = Path(data_root_dir)/'Raw'/year/experiment/location/population/date/platform/sensor/'Metadata'
        save = Path(data_root_dir)/'Processed'/year/experiment/location/population/date/platform/sensor/f'{date}-{platform}-{sensor}-Traits-WGS84.geojson'
        save_extract = Path(data_root_dir)/'Processed'/year/experiment/location/population/date/platform/sensor
        temp = Path(data_root_dir)/'Processed'/year/experiment/location/population/date/platform/sensor/'temp'
        temp_extract = Path(data_root_dir)/'Processed'/year/experiment/location/population/date/platform/sensor/'temp'
        temp_extract.mkdir(parents=True, exist_ok=True) #if it doesnt exists
        save_extract.mkdir(parents=True, exist_ok=True)
        
        # reset extract process (or initialize)
        extract_process = None
        
        # check if date is emerging
        emerging = date in summary
        
        # check if metadata path exists OR contains files
        if not metadata.exists() or not any(metadata.iterdir()):
            return jsonify({"error": "Platform logs not found or empty. Please press stop and upload necessary logs."}), 404
        
        # run extract
        if emerging:
            if disparity.exists():
                cmd = (
                    f"python -W ignore {EXTRACT_TRAITS} "
                    f"--emerging --summary '{summary_path}' --images '{images}' --plotmap '{plotmap}' "
                    f"--batch-size {batch_size} --model-path '{model_path}' --save '{save_extract}' "
                    f"--metadata '{metadata}' --temp '{temp}' --trait '{trait}' --skip-stereo --geojson-filename '{save}'"
                )
            else:
                cmd = (
                    f"python -W ignore {EXTRACT_TRAITS} "
                    f"--emerging --summary '{summary_path}' --images '{images}' --plotmap '{plotmap}' "
                    f"--batch-size {batch_size} --model-path '{model_path}' --save '{save}' "
                    f"--metadata '{metadata}' --temp '{temp}' --trait '{trait}' --geojson-filename '{save}'"
                )
        else:
            if disparity.exists():
                cmd = (
                    f"python -W ignore {EXTRACT_TRAITS} "
                    f"--summary '{summary_path}' --images '{images}' --plotmap '{plotmap}' "
                    f"--batch-size {batch_size} --model-path '{model_path}' --save '{save}' "
                    f"--metadata '{metadata}' --temp '{temp}' --trait '{trait}' --skip-stereo --geojson-filename '{save}'"
                )
            else:
                cmd = (
                    f"python -W ignore {EXTRACT_TRAITS} "
                    f"--summary '{summary_path}' --images '{images}' --plotmap '{plotmap}' "
                    f"--batch-size {batch_size} --model-path '{model_path}' --save '{save}' "
                    f"--metadata '{metadata}' --temp '{temp}' --trait '{trait}' --geojson-filename '{save}'"
                )
        print(cmd)
        
        extract_process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        threading.Thread(target=stream_output, args=(extract_process,), daemon=True).start()
        time.sleep(5)  # Wait for 5 seconds
        if extract_process.poll() is None:
            print("Extract process started successfully and is running.")
            return jsonify({"message": "Extract started"}), 202
        else:
            print("Extract process failed to start or exited immediately.")
            return jsonify({"error": 
                "Failed to start extraction process. Check if you have corectly uploaded images/metadata"}), 404
    
    except subprocess.CalledProcessError as e:
        error_output = e.stderr.decode('utf-8')
        return jsonify({"status": "error", "message": str(error_output)}), 404
    
@model_bp.route('/stop_extract', methods=['POST'])
def stop_extract():
    global save_extract, temp_extract, extract_process
    try:
        print('Extract stopped by user.')
        if extract_process is not None:
            extract_process.terminate()
            extract_process.wait()
            print("Extract process terminated.")
            extract_process = None
        else:
            print("No extract process running.")
        
        subprocess.run(f"rm -rf '{save_extract}/logs.yaml'", check=True, shell=True)
        subprocess.run(f"rm -rf '{temp_extract}'", check=True, shell=True)
        return jsonify({"message": "Python process successfully stopped"}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": e.stderr.decode("utf-8")}), 500

@model_bp.route('/done_extract', methods=['POST'])
def done_extract():
    global temp_extract, save_extract, model_id, summary_date, locate_id, trait_extract, extract_process
    try:
        # update logs file
        logs_file = Path(save_extract)/'logs.yaml'
        if logs_file.exists():
            with open(logs_file, 'r') as file:
                data = yaml.safe_load(file) or {} # use an empty dict if the file is empty
        else:
            data = {}
        new_values = {
            "model": model_id,
            "locate": summary_date,
            "id": locate_id
        }
        update_or_add_entry(data, trait_extract, new_values)
        with open(logs_file, 'w') as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)
        
        print('Extract stopped by user.')
        if extract_process is not None:
            extract_process.terminate()
            extract_process.wait()
            print("Extract process terminated.")
            extract_process = None
        else:
            print("No extract process running.")
        subprocess.run(f"rm -rf '{temp_extract}'", check=True, shell=True)
        return jsonify({"message": "Python process in container successfully stopped"}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": e.stderr.decode("utf-8")}), 500