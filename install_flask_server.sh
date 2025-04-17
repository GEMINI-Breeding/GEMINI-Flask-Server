#!/usr/bin/env bash

# Function to download and install Miniconda
install_miniconda() {
    echo "Conda is not installed. Downloading and installing Miniconda..."

    ARCH=$(uname -m)
    if [[ "$ARCH" == "arm64" ]]; then
        # Apple Silicon
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
    else
        # Intel
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
    fi

    MINICONDA_SCRIPT=$(basename "$MINICONDA_URL")

    # Download the Miniconda installer
    curl -LO "$MINICONDA_URL"

    # Run the Miniconda installer
    bash "$MINICONDA_SCRIPT" -b -p "$HOME/miniconda"

    # Initialize conda
    "$HOME/miniconda/bin/conda" init

    # Remove the installer script
    rm "$MINICONDA_SCRIPT"

    # Source the new conda configuration (detects shell type)
    if [[ "$SHELL" == *zsh ]]; then
        source "$HOME/.zshrc"
    else
        source "$HOME/.bash_profile"
    fi
}

# Check if Conda is installed
if ! command -v conda &> /dev/null; then
    install_miniconda
else
    echo "Conda is already installed."
fi

# Check if the script is running on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Running on macOS."
    export CXXFLAGS="-std=c++11"
    
    # Check if Xcode is installed
    if ! xcode-select -p &> /dev/null; then
        echo "Xcode is not installed. Please install Xcode via the App Store before proceeding."
        exit 1
    else
        echo "Xcode is installed."
    fi

    # Check if pcregrep is available
    if ! command -v pcregrep &> /dev/null; then
        echo "Error: pcregrep is not installed. Please run 'brew install pcre' and try again."
        exit 1
    fi
fi

# Check if conda environment exists, if not, create it.
if [ ! -d ".conda" ]; then
    # Create conda env
    echo "Creating conda environment..."
    conda env create -f gemini-flask-server.yml -p .conda
else
    echo ".conda already exists."
    read -p "Do you want to reinstall the conda environment? (y/n): " choice
    case "$choice" in 
      y|Y ) 
        echo "Reinstalling conda environment..."
        conda env remove -p .conda
        conda env create -f gemini-flask-server.yml -p .conda
        ;;
      n|N ) 
        echo "Keeping the existing conda environment."
        ;;
      * ) 
        echo "Invalid option. Exiting."
        ;;
    esac
fi

# Initialize conda for the current shell session
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ./.conda

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Clone the farm-ng-core repo
    git clone https://github.com/farm-ng/farm-ng-core.git

    # Checkout the correct release and update submodules
    cd farm-ng-core/
    git checkout v2.3.0
    git submodule update --init --recursive
    cd ../

    # Upgrade pip and setuptools using conda's pip
    echo "Upgrading pip and setuptools in Conda environment..."
    ./.conda/bin/pip install --upgrade pip setuptools wheel

    # Build farm-ng-core from source using conda's pip
    echo "Installing farm-ng-core with Conda pip..."
    cd farm-ng-core/
    ../.conda/bin/pip install .
    cd ../

    # Install farm-ng-amiga using farm-ng-core
    echo "Installing farm-ng-amiga with Conda pip..."
    ./.conda/bin/pip install --no-build-isolation farm-ng-amiga
else
    pip3 install farm-ng-amiga
fi