#!/usr/bin/env bash
# Echo current directory
echo "Current directory: $(pwd)"

# Change to the GEMINI-Flask-Server directory
pushd "$(pwd)/../GEMINI-Flask-Server"

# Resolve the full absolute path to the .conda environment
conda_env_path="$(cd ./.conda && pwd)"
echo "Resolved Conda environment path: $conda_env_path"

# Initialize Conda in the current shell session
echo "Sourcing conda from: $(conda info --base)/etc/profile.d/conda.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate the Conda environment by path
echo "Activating Conda environment at: $conda_env_path"
conda activate "$conda_env_path"

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
# echo "data_root_dir: $data_root_dir"

# Read the port from the arguments
if [ -z "$2" ]; then
    flask_port=5000 # Default port
else
    flask_port=$2
fi
# echo "Flask Port: $flask_port"

# Read the port from the arguments
if [ -z "$3" ]; then
    titiler_port=8091 # Default port
else
    titiler_port=$3
fi
# echo "Titiler Port: $titiler_port"

python src/app_flask_backend.py --data_root_dir $data_root_dir --flask_port $flask_port --titiler_port $titiler_port

# Change back to the original directory
popd