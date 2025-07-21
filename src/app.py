# Standard library imports
import os
import argparse
import subprocess
import multiprocessing

# Third-party library imports
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.middleware.cors import CORSMiddleware

# Import flask app
from app_flask_backend import file_app

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir', type=str, default='~/GEMINI-App-Data', required=False)
    parser.add_argument('--flask_port', type=int, default=5000, required=False)
    parser.add_argument('--titiler_port', type=int, default=8091, required=False)
    args = parser.parse_args()

    print(f"flask_port: {args.flask_port}")
    print(f"titiler_port: {args.titiler_port}")
    data_root_dir = args.data_root_dir
    if "~" in data_root_dir:
        data_root_dir = os.path.expanduser(data_root_dir)
    print(f"data_root_dir: {data_root_dir}")
    file_app.config['DATA_ROOT_DIR'] = data_root_dir
    file_app.config['UPLOAD_BASE_DIR'] = os.path.join(data_root_dir, 'Raw')
    file_app.config['NOW_DRONE_PROCESSING'] = False
    app.mount("/flask_app", WSGIMiddleware(file_app))
    titiler_command = f"uvicorn titiler.application.main:app --reload --port {args.titiler_port}"
    titiler_process = subprocess.Popen(titiler_command, shell=True)
    uvicorn.run(app, host="127.0.0.1", port=args.flask_port)
    titiler_process.terminate()