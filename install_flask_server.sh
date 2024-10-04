#!/usr/bin/env bash

# Function to download and install Miniconda
install_miniconda() {
    echo "Conda is not installed. Downloading and installing Miniconda..."
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
    MINICONDA_SCRIPT="Miniconda3-latest-MacOSX-x86_64.sh"
    
    # Download the Miniconda installer
    curl -LO $MINICONDA_URL
    
    # Run the Miniconda installer
    bash $MINICONDA_SCRIPT -b -p $HOME/miniconda
    
    # Initialize conda
    $HOME/miniconda/bin/conda init
    
    # Remove the installer script
    rm $MINICONDA_SCRIPT
    
    # Source the new conda configuration
    source $HOME/.bash_profile  # or ~/.zshrc, depending on the shell you're using
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

    # [Optional] Upgrade some deps
    pip3 install --upgrade pip
    pip3 install --upgrade setuptools wheel

    # Build farm-ng-core from source
    cd farm-ng-core/
    pip3 install .
    cd ../

    # Install farm-ng-amiga wheel, using farm-ng-core built from source
    pip3 install --no-build-isolation farm-ng-amiga
else
   pip3 install farm-ng-amiga
fi