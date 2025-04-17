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
echo "data_root_dir: $data_root_dir"

# Read the port from gemini-app/src/DataContext.js
if [[ "$(uname)" == "Darwin" ]]; then
    # Use pcregrep on macOS
    port=$(pcregrep -o1 'const \[flaskUrl, setFlaskUrl\] = useState\("http://127.0.0.1:\K([0-9]+)' ../gemini-app/src/DataContext.js)
else
    # Use grep -P on other systems
    port=$(grep -oP 'const \[flaskUrl, setFlaskUrl\] = useState\("http://127.0.0.1:\K[0-9]+' ../gemini-app/src/DataContext.js)
fi
echo "Flask Port: $port"

python src/app_flask_backend.py --data_root_dir $data_root_dir --port $port

# Change back to the original directory
popd