#!/usr/bin/env bash
# echo current directory
echo "Current directory: $(pwd)"
# Change to the directory of GEMINI-Flask-Server
pushd $(pwd)/../GEMINI-Flask-Server 
# bash ./install_flask_server.sh

# Read conda path from your ~/.bashrc and parse the conda path before :$PATH
if [[ "$(uname)" == "Darwin" ]]; then
    conda_path=$(pcregrep -o '__conda_setup="\$\(\K[^:]+/bin' ~/.zshrc | awk -F':' '{print $1}' | tr -d "'")
else
    conda_path=$(grep -oP 'export PATH="\K[^:]+/bin' ~/.bashrc)
fi

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
        data_root_dir="/home/GEMINI/GEMINI-App-Data"
    fi
else
    data_root_dir=$1
fi
echo "data_root_dir: $data_root_dir"

# Read the port from the arguments
if [ -z "$2" ]; then
    flask_port=5050 # Default port
else
    flask_port=$2
fi
echo "Flask Port: $flask_port"

# Read the port from the arguments
if [ -z "$3" ]; then
    titiler_port=8091 # Default port
else
    titiler_port=$3
fi
echo "Titiler Port: $titiler_port"

python src/app_flask_backend.py --data_root_dir $data_root_dir --flask_port $flask_port --titiler_port $titiler_port

# Change back to the original directory
popd