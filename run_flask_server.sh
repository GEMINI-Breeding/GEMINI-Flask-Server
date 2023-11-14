#!/usr/bin/env bash
pushd ../GEMINI-Flask-Server # Change to the directory of GEMINI-Flask-Server
# bash ./install_flask_server.sh

# Activate conda env
# source activate .conda/
source activate ~/miniconda3/envs/gemini-flask-server

# Use default arguments when they are not provided (data_root_dir, port)
if [ -z "$1" ]; then
    data_root_dir="/home/GEMINI/GEMINI-Data"
else
    data_root_dir=$1
fi

if [ -z "$2" ]; then
    port=5050
else
    port=$2
fi

python src/app_flask_backend.py --data_root_dir $data_root_dir --port $port

# Change back to the original directory
popd