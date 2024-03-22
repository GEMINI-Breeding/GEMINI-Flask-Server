#!/usr/bin/env bash

# echo current directory
echo "Current directory: $(pwd)"

# Change to the directory of GEMINI-Flask-Server
pushd $(pwd)/../GEMINI-Flask-Server 

# Read conda path from your ~/.bashrc and parse the conda path before :$PATH
conda_path=$(grep -oP 'export PATH="\K[^:]+/bin' ~/.bashrc)
echo "Conda Path: $conda_path"
# Activate conda environment
source $conda_path/activate .conda/

# Use default arguments when they are not provided (data_root_dir, port)
if [ -z "$1" ]; then
    # Check if the data_root_dir exists
    data_root_dir="/home/gemini/data/GEMINI-App-Data"
    if [ ! -d "$data_root_dir" ]; then
        echo "The data_root_dir does not exist: $data_root_dir"
        # Try default data_root_dir
        data_root_dir="/home/GEMINI/GEMINI-Data"
    fi
else
    data_root_dir=$1
fi
echo "data_root_dir: $data_root_dir"

# Read the port from gemini-app/src/DataContext.js
port=$(grep -oP 'const \[flaskUrl, setFlaskUrl\] = useState\("http://127.0.0.1:\K[0-9]+' ../gemini-app/src/DataContext.js)
echo "Flask Port: $port"

python src/app_flask_backend.py --data_root_dir $data_root_dir --port $port

# Change back to the original directory
popd