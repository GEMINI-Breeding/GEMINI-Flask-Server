# Standard library imports
import os
import subprocess
import threading
import traceback
import argparse
# Third-party library imports

import uvicorn
from flask import Flask

# Import inference module
from scripts.roboflow_inference import register_inference_routes
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.middleware.cors import CORSMiddleware

# Blueprint imports
from blueprints.plot_marking import plot_marking_bp
from blueprints.file_management import file_management_bp
from blueprints.upload_management import upload_management_bp
from blueprints.processing import processing_bp
from blueprints.postprocessing import postprocessing_bp
from blueprints.trait_processing import model_bp

# stitch pipeline
import sys
AGROWSTITCH_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../AgRowStitch"))
print(AGROWSTITCH_PATH)
sys.path.append(AGROWSTITCH_PATH)
from scripts.directory_index import DirectoryIndex, DirectoryIndexDict

# Define the Flask application for serving files
file_app = Flask(__name__)
file_app.register_blueprint(plot_marking_bp)
file_app.register_blueprint(file_management_bp)
file_app.register_blueprint(upload_management_bp)
file_app.register_blueprint(processing_bp)
file_app.register_blueprint(postprocessing_bp)
file_app.register_blueprint(model_bp)


latest_data = {'epoch': 0, 'map': 0, 'locate': 0, 'extract': 0, 'ortho': 0, 'drone_extract': 0}
training_stopped_event = threading.Event()
extraction_processes = {}
extraction_status = "not_started"  # Possible values: not_started, in_progress, done, failed
extraction_error_message = None  # Stores detailed error message if extraction fails
odm_method = None
stitch_thread=None
stitch_stop_event = threading.Event()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/flask_app", WSGIMiddleware(file_app))

if __name__ == "__main__":
    
    # Add arguments to the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir', type=str, default='~/GEMINI-App-Data',required=False)
    parser.add_argument('--flask_port', type=int, default=5000,required=False) # Default port is 5000
    parser.add_argument('--titiler_port', type=int, default=8091,required=False) # Default port is 8091
    args = parser.parse_args()

    # Print the arguments to the console
    print(f"flask_port: {args.flask_port}")
    print(f"titiler_port: {args.titiler_port}")

    # Update global data_root_dir from the argument
    global data_root_dir
    data_root_dir = args.data_root_dir
    if "~" in data_root_dir:
        data_root_dir = os.path.expanduser(data_root_dir)
    print(f"data_root_dir: {data_root_dir}")
    file_app.config['DATA_ROOT_DIR'] = data_root_dir
    global UPLOAD_BASE_DIR
    UPLOAD_BASE_DIR = os.path.join(data_root_dir, 'Raw')
    file_app.config['UPLOAD_BASE_DIR'] = UPLOAD_BASE_DIR
    global dir_db
    if 1:
        db_path = os.path.join(data_root_dir, "directory_index_dict.pkl")
        dir_db = None
        # Use dictionary-based index
        dir_db = DirectoryIndexDict(verbose=False)
        # Try loading from file if exists
        if os.path.exists(db_path):
            dir_db.load_dict(db_path)
            print(f"Loaded directory index dict from {db_path}")
        else:
            print(f"No dict file found, will build index from scratch.")
    else:
        db_path = os.path.join(data_root_dir, "directory_index.db")
        dir_db = None
        # Use SQLite-based index
        dir_db = DirectoryIndex(db_path=db_path, verbose=False)
        # No need to load_dict or save_dict for DirectoryIndex

    # Register inference routes
    file_app.config['DIR_DB'] = dir_db
    file_app.config['DB_PATH'] = db_path

    file_app.config['LATEST_DATA'] = latest_data
    file_app.config['TRAINING_STOPPED_EVENT'] = training_stopped_event
    file_app.config['EXTRACTION_PROCESSES'] = extraction_processes
    file_app.config['EXTRACTION_STATUS'] = extraction_status
    file_app.config['EXTRACTION_ERROR_MESSAGE'] = extraction_error_message
    file_app.config['ODM_METHOD'] = odm_method
    file_app.config['STITCH_THREAD'] = stitch_thread
    file_app.config['STITCH_STOP_EVENT'] = stitch_stop_event

    register_inference_routes(file_app, data_root_dir)

    global now_drone_processing
    now_drone_processing = False
    file_app.config['NOW_DRONE_PROCESSING'] = now_drone_processing

    # Start the Titiler server using the subprocess module
    titiler_command = f"uvicorn titiler.application.main:app --reload --port {args.titiler_port}"
    titiler_process = subprocess.Popen(titiler_command, shell=True)

    # Start the Flask server
    uvicorn.run(app, port=args.flask_port)

    # Save the directory index dict before shutdown
    if ".pkl" in db_path:
        dir_db.save_dict(db_path)

    # Terminate the Titiler server when the Flask server is shut down
    titiler_process.terminate()