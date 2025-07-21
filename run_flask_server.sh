#!/usr/bin/env bash
set -euo pipefail

# Echo current directory
echo "Current directory: $(pwd)"

# Change to the GEMINI-Flask-Server directory
pushd "$(pwd)/../GEMINI-Flask-Server"

# Resolve the full absolute path to the .conda environment
conda_env_path="$(cd ./.conda && pwd)"
echo "Resolved Conda environment path: $conda_env_path"

# Common Conda installation directories to check
POSSIBLE_CONDA_DIRS=(
    "$HOME/miniconda3"
    "$HOME/anaconda3"
    "/opt/conda"
    "/usr/local/miniconda3"
    "/usr/local/anaconda3"
)

echo "Checking for Conda installations in the following paths:"
for dir in "${POSSIBLE_CONDA_DIRS[@]}"; do
    echo "  - $dir"
    if [ -f "$dir/etc/profile.d/conda.sh" ]; then
        echo "✅ Found Conda at: $dir"
        source "$dir/etc/profile.d/conda.sh"
        CONDA_FOUND=true
        break
    fi
done

# Activate the environment using resolved path (add a check)
echo "Activating Conda environment at: $conda_env_path"
source activate "$conda_env_path" || {
    echo "❌ Error: Failed to activate Conda environment at $conda_env_path"
    exit 1
}
echo "✅ Conda environment activated successfully."

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
flask_port="${2:-5000}"
echo "Flask Port: $flask_port"

# Read the titiler_port from the arguments, with default
titiler_port="${3:-8091}"
echo "Titiler Port: $titiler_port"

python src/app.py --data_root_dir $data_root_dir --flask_port $flask_port --titiler_port $titiler_port

# Change back to the original directory
popd