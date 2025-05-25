#!/bin/bash

PYTHON_CMD="python3"
VENV_NAME="venv_tstorm"
REQUIREMENTS_FILE="requirements.txt"
PARAMETERS_FILE="parameters.txt"
MAIN_SCRIPT="main.py"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )" # Get script's own directory

cd "$SCRIPT_DIR"

if ! command -v $PYTHON_CMD &> /dev/null
then
    echo "ERROR: $PYTHON_CMD is not installed or not found in your PATH."
    echo "Please install Python 3 (e.g., from python.org or using your system's package manager) to continue."
    exit 1
fi

# Create a virtual environment if it doesn't exist or if requirements changed
VENV_DIR="$SCRIPT_DIR/$VENV_NAME"
SETUP_TRIGGER_FILE="$VENV_DIR/.setup_complete_trigger"

# Simplified check: if venv exists and trigger file exists, assume setup is okay.
PERFORM_SETUP=false
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment '$VENV_NAME' not found."
    PERFORM_SETUP=true
elif [ ! -f "$SETUP_TRIGGER_FILE" ]; then
    echo "Virtual environment '$VENV_NAME' found, but setup trigger file is missing."
    PERFORM_SETUP=true
fi


if [ "$PERFORM_SETUP" = true ] ; then
    echo "Creating/Re-initializing Python virtual environment in '$VENV_DIR'..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment."
        exit 1
    fi

    echo "Activating virtual environment for setup..."
    source "$VENV_DIR/bin/activate"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to activate virtual environment for setup."
        exit 1
    fi

    echo "Upgrading pip..."
    pip install --upgrade pip
    
    echo "Installing required Python packages from $REQUIREMENTS_FILE..."
    pip install -r "$REQUIREMENTS_FILE"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install Python packages."
        echo "Please check your internet connection and $REQUIREMENTS_FILE."
        echo "If you encounter issues with 'geopandas', you might need to install its system dependencies first."
        echo "  Common dependencies: gdal, geos, proj."
        echo "  On Debian/Ubuntu: sudo apt-get install libgdal-dev gdal-bin python3-gdal"
        echo "  On macOS with Homebrew: brew install gdal geos proj"
        echo "The simulation can run without geopandas (map boundaries will be skipped)."
        deactivate
        exit 1
    fi
    touch "$SETUP_TRIGGER_FILE" # Mark setup as complete
    echo "Packages installed successfully."
    deactivate
    echo "Virtual environment setup complete and deactivated. Will reactivate for run."
else
    echo "Virtual environment '$VENV_NAME' appears to be already set up."
fi

# Activate the virtual environment for running the script
echo "Activating virtual environment for run: $VENV_DIR/bin/activate"
source "$VENV_DIR/bin/activate"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment for run."
    exit 1
fi


# Check if parameters.txt exists
if [ ! -f "$PARAMETERS_FILE" ]; then
    echo "ERROR: '$PARAMETERS_FILE' not found in $SCRIPT_DIR!"
    echo "Please create it. You can copy the example from the README.md file."
    deactivate
    exit 1
fi

echo ""
echo "----------------------------------------------------"
echo "IMPORTANT: Please ensure you have edited the '$PARAMETERS_FILE' file"
echo "with your desired simulation inputs before proceeding."
echo "----------------------------------------------------"
echo ""
# Add a small pause or a direct prompt
read -p "Press [Enter] to run the simulation, or Ctrl+C to abort and edit parameters..."

# Execute the Python script
$PYTHON_CMD "$MAIN_SCRIPT"
EXIT_CODE=$?
#$SCRIPT_DIR
# echo ""
# echo "----------------------------------------------------"
# if [ $EXIT_CODE -eq 0 ]; then
#     echo "Simulation script completed successfully."
#     echo "Output plots: afternoon_thunderstorm_simulation.png, cloud_wind_simulation.png"
# else
#     echo "Python script exited with error code $EXIT_CODE."
#     echo "Please check the error messages above."
# fi
# echo "===================================================="

# Deactivate the virtual environment
deactivate