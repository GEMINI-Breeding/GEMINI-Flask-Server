#!/usr/bin/env bash
pushd /home/gemini/scratch/GEMINI-Flask-Server # Change to the directory of GEMINI-Flask-Server
# bash ./install_flask_server.sh

# Activate conda env
source /home/gemini/miniconda3/bin/activate .conda/

# Use default arguments when they are not provided (data_root_dir, port)
if [ -z "$1" ]; then
    data_root_dir="/home/gemini/data/GEMINI-App-Data"
else
    data_root_dir=$1
fi
echo $data_root_dir

if [ -z "$2" ]; then
    port=5003
else
    port=$2
fi

python src/app_flask_backend.py --data_root_dir $data_root_dir --port $port

# Change back to the original directory
popd