#!/usr/bin/env bash
# Check if conda env is exist, if not, create it.
if [ ! -d ".conda" ]; then
    # Create conda env
    echo "Create conda env"
    conda env create -f gemini-flask-server.yml -p .conda
else
    echo ".conda already exist"
fi
