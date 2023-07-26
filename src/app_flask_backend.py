import os
import subprocess
import threading
import uvicorn

from flask import Flask, send_from_directory, jsonify, request
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.middleware.cors import CORSMiddleware


# Define the Flask application for serving files
file_app = Flask(__name__)

#### FILE SERVING ENDPOINTS ####
# endpoint to serve files
@file_app.route('/files/<path:filename>')
def serve_files(filename):
    return send_from_directory('/home/GEMINI/GEMINI-Data', filename)

# endpoint to list files
@file_app.route('/list_dirs/<path:dir_path>', methods=['GET'])
def list_dirs(dir_path):
    dir_path = os.path.join('/home/GEMINI/GEMINI-Data', dir_path)  # join with base directory path
    if os.path.exists(dir_path):
        dirs = next(os.walk(dir_path))[1]
        return jsonify(dirs), 200
    else:
        return jsonify({'message': 'Directory not found'}), 404

#### SCRIPT SERVING ENDPOINTS ####
# endpoint to run script
@file_app.route('/run_script', methods=['POST'])
def run_script():
    data = request.json
    script_path = data.get('script_path')

    def run_in_thread(script_path):
        subprocess.call(script_path, shell=True)

    thread = threading.Thread(target=run_in_thread, args=(script_path,))
    thread.start()

    return jsonify({'message': 'Script started'}), 200

# FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Flask app to FastAPI
app.mount("/flask_app", WSGIMiddleware(file_app))

# Add Titiler to FastAPI
# app.mount("/cog", app=titiler_app, name='titiler')

if __name__ == "__main__":

    # Start the Titiler server using the subprocess module
    titiler_command = "uvicorn titiler.application.main:app --reload --port 8090"
    titiler_process = subprocess.Popen(titiler_command, shell=True)

    # Start the Flask server
    uvicorn.run(app, host="0.0.0.0", port=5000)

    # Terminate the Titiler server when the Flask server is shut down
    titiler_process.terminate()
