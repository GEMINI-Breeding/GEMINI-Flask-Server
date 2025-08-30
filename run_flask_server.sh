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

# Function to check if environment needs updating
check_environment_update() {
    echo "Checking if conda environment needs updating..."
    
    # Get modification times
    local conda_env_time=0
    local requirements_time=0
    local yml_time=0
    
    # Check .conda directory modification time - try multiple files to find the most recent
    if [ -d "$conda_env_path" ]; then
        # Try multiple files to get the environment creation/modification time
        local temp_time=0
        
        # Check pyvenv.cfg
        if [ -f "$conda_env_path/pyvenv.cfg" ]; then
            temp_time=$(stat -c %Y "$conda_env_path/pyvenv.cfg" 2>/dev/null || echo 0)
            [ $temp_time -gt $conda_env_time ] && conda_env_time=$temp_time
        fi
        
        # Check conda-meta directory (contains package info)
        if [ -d "$conda_env_path/conda-meta" ]; then
            temp_time=$(stat -c %Y "$conda_env_path/conda-meta" 2>/dev/null || echo 0)
            [ $temp_time -gt $conda_env_time ] && conda_env_time=$temp_time
        fi
        
        # Check lib directory
        if [ -d "$conda_env_path/lib" ]; then
            temp_time=$(stat -c %Y "$conda_env_path/lib" 2>/dev/null || echo 0)
            [ $temp_time -gt $conda_env_time ] && conda_env_time=$temp_time
        fi
        
        # Check bin directory
        if [ -d "$conda_env_path/bin" ]; then
            temp_time=$(stat -c %Y "$conda_env_path/bin" 2>/dev/null || echo 0)
            [ $temp_time -gt $conda_env_time ] && conda_env_time=$temp_time
        fi
        
        # If still 0, use the .conda directory itself
        if [ $conda_env_time -eq 0 ]; then
            conda_env_time=$(stat -c %Y "$conda_env_path" 2>/dev/null || echo 0)
        fi
    fi

    
    # Check requirements.txt
    if [ -f "requirements.txt" ]; then
        requirements_time=$(stat -c %Y "requirements.txt" 2>/dev/null || echo 0)
    fi
    
    # Check gemini-flask-server.yml
    if [ -f "gemini-flask-server.yml" ]; then
        yml_time=$(stat -c %Y "gemini-flask-server.yml" 2>/dev/null || echo 0)
    fi
    
    echo "Environment creation time: $(date -d @$conda_env_time 2>/dev/null || echo 'Unknown')"
    echo "Requirements.txt time: $(date -d @$requirements_time 2>/dev/null || echo 'Not found')"
    echo "Gemini-flask-server.yml time: $(date -d @$yml_time 2>/dev/null || echo 'Not found')"
    
    # Check if either requirements.txt or yml file is newer than the conda environment
    if [ $requirements_time -gt $conda_env_time ] || [ $yml_time -gt $conda_env_time ]; then
        echo "🔄 Environment update needed - dependency files are newer than conda environment"
        return 0
    else
        echo "✅ Environment is up to date"
        return 1
    fi
}

# Update environment if needed
update_environment() {
    echo "Updating conda environment..."
    
    # First try to update from yml file if it exists and is newer
    if [ -f "gemini-flask-server.yml" ]; then
        echo "Updating environment from gemini-flask-server.yml..."
        # Use --prefix to update the current .conda environment instead of creating new one
        conda env update --prefix "$conda_env_path" --file gemini-flask-server.yml --prune || {
            echo "⚠️  Warning: Failed to update from yml file, trying requirements.txt"
        }
    fi
    
    echo "✅ Environment update completed"
}

# Check and update environment if needed
if check_environment_update; then
    update_environment
fi

# Deactivate any currently active conda environment
if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
    echo "Deactivating currently active conda environment: $CONDA_DEFAULT_ENV"
    conda deactivate
fi

# Activate the environment using resolved path (add a check)
echo "Activating Conda environment at: $conda_env_path"
conda activate "$conda_env_path" || {
    echo "❌ Error: Failed to activate Conda environment at $conda_env_path"
    exit 1
}
echo "✅ Conda environment activated successfully."

# Use default arguments when they are not provided (data_root_dir, port)
if [ $# -eq 0 ] || [ -z "${1:-}" ]; then
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

python src/app_flask_backend.py --data_root_dir $data_root_dir --flask_port $flask_port --titiler_port $titiler_port

# Change back to the original directory
popd