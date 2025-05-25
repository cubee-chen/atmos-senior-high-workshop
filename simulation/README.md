# Taipei Afternoon Thunderstorm Simulation

This program simulates a conceptual afternoon thunderstorm over the Taipei Basin.
You can change various weather parameters to see how they affect the simulation outcome, particularly the rainfall at the NTU (National Taiwan University) station.

## Prerequisites

1.  **Python 3:** You need Python 3 installed on your computer.
    * **Windows:** Download from [python.org](https://www.python.org/downloads/). **IMPORTANT**: During installation, make sure to check the box that says "Add Python to PATH" or "Add python.exe to PATH".
    * **macOS:** Python 3 might already be installed. Open Terminal (Applications > Utilities > Terminal) and type `python3 --version`. If not found or it's an old version (e.g., Python 2), install Python 3 from [python.org](https://www.python.org/downloads/) or using Homebrew (`brew install python`).
    * **Linux:** Use your distribution's package manager (e.g., `sudo apt update && sudo apt install python3 python3-venv python3-pip` on Debian/Ubuntu).

2.  **Shapefile Data (Optional, for map boundaries):**
    * The `DATA` folder in this package should contain the town boundary shapefile (e.g., `TOWN_MOI_1131028.shp` and its associated files like `.dbf`, `.shx`, `.prj`, `.cpg`). This is used to draw map boundaries on the output plots.
    * If this folder or its contents are missing, or if the `geopandas` library cannot be installed (see below), the simulation will still run, but the map boundaries will not be shown.

## How to Run the Simulation

**First-time setup and subsequent runs:**

1.  **Unzip:**
    * Download the `afternoon_thunderstorm_simulation.zip` file.
    * Unzip it into a new folder on your computer (e.g., on your Desktop or in your Documents folder). Let's call this folder `MyStormSim`.

2.  **Edit Parameters (This is where you control the weather!):**
    * Inside your `MyStormSim` folder, find the file named `parameters.txt`.
    * Open `parameters.txt` with a simple text editor:
        * **Windows:** Notepad (search for Notepad in the Start Menu).
        * **macOS:** TextEdit (Applications > TextEdit). **Important for TextEdit:** Go to Format > Make Plain Text if it's not already.
        * You can also use more advanced editors like VS Code, Sublime Text, etc.
    * Read the comments (lines starting with `#`) in `parameters.txt` to understand what each parameter does.
    * **Change the values** for parameters like `solar_radiation`, `initial_cloud_coverage`, `average_wind_speed`, temperatures, etc.
    * **Save the `parameters.txt` file after making your changes.**

3.  **Run the Simulation Script:**
    * **Windows:**
        1.  Navigate into your `MyStormSim` folder.
        2.  Double-click the `run_simulation.bat` file. A command prompt window will open.
    * **macOS/Linux:**
        1.  Open the "Terminal" application.
        2.  Navigate to your `MyStormSim` folder. For example, if it's on your Desktop:
            ```bash
            cd Desktop/MyStormSim
            ```
        3.  Make the script executable (you only need to do this once):
            ```bash
            chmod +x run_simulation.sh
            ```
        4.  Run the script:
            ```bash
            ./run_simulation.sh
            ```

4.  **Follow Prompts:**
    * **First Run:** The script will first try to set up a Python "virtual environment" (a private workspace for this project) and install the necessary Python libraries (like `numpy`, `matplotlib`, `geopandas`). This might take a few minutes, especially if `geopandas` needs to be compiled or downloaded. You'll need an internet connection for this first-time setup.
        * *(Troubleshooting `geopandas`):* The `geopandas` library is used for drawing map boundaries. It can sometimes be tricky to install because it depends on other system libraries (GDAL, GEOS, PROJ).
            * If `geopandas` fails to install, the script will print a warning. **The simulation can still run without it, but you won't see the town boundaries on the map.**
            * The script might give some hints on how to install these dependencies if you're an advanced user. For most students, it's okay to proceed without `geopandas` if it's problematic.
    * **Subsequent Runs:** If the setup was completed before, the script will skip the installation part and run faster.
    * The script will then ask you to press a key (Enter or X) to confirm you're ready to run with the current `parameters.txt`.

5.  **View Results:**
    * The simulation will run in the terminal/command prompt window, printing out information about the parameters and results.
    * When finished, two image files will be created in your `MyStormSim` folder:
        * `afternoon_thunderstorm_simulation.png`: Shows the simulated precipitation map.
        * `cloud_wind_simulation.png`: Shows the simulated cloud cover and wind field.
    * Open these images to see your simulation's weather!
    * The terminal will also tell you the simulated precipitation at NTU and whether it exceeded a predefined threshold.

6.  **Run Again with New Parameters:**
    * To run the simulation with different weather settings:
        1.  Re-open and edit `parameters.txt` in your `MyStormSim` folder.
        2.  Save your changes to `parameters.txt`.
        3.  Run `run_simulation.bat` (Windows) or `./run_simulation.sh` (macOS/Linux) again.

## Understanding `parameters.txt` (Your Weather Control Panel!)

This file is where you tell the simulation what the starting weather conditions are.

* `solar_radiation` (Scale 1-10): How strong the sun is. Higher values usually mean more energy for storms.
* `initial_cloud_coverage` (Fraction 0.0-1.0): How cloudy it is at the start. `0.0` means clear sky, `1.0` means fully overcast. Less cloud cover initially can lead to more surface heating and potentially stronger storms.
* `average_wind_speed` (m/s): The general wind speed.
* `wind_direction_grid`: This defines the wind direction (in degrees, where 0/360 is North, 90 is East, 180 is South, 270 is West) over different parts of the simulation area. It's a 3x3 grid represented as a Python list of lists.
    * The format is `[[NW_corner, N_edge, NE_corner], [W_edge, Center, E_edge], [SW_corner, S_edge, SE_corner]]`.
    * Example for a mostly South-Westerly flow: `[[240,235,230],[235,225,220],[230,220,215]]`
* `moisture_availability` (Scale 1-10): How much moisture is in the air. More moisture means more potential for rain.
* `near_surface_temp_c`: Temperature near the ground in Celsius (e.g., `32.0`).
* `upper_air_temp_c`: Temperature higher up in the atmosphere in Celsius (e.g., `-5.0`). A large difference between surface and upper air temperature (hot surface, cold air aloft) often leads to more "atmospheric instability" and stronger storms.
* `random_seed`:
    * If you set this to an integer number (e.g., `42`), the simulation will produce the *exact same* random patterns every time you run it with that seed (as long as other parameters are also the same). This is useful for testing specific changes.
    * If you leave it blank or type `None` (without quotes), the simulation will use a different set of random numbers each time, leading to slightly different storm formations even with the same weather parameters. This shows the natural variability in weather.
* `town_shapefile_path`: Path to the map boundary file. You usually don't need to change this if you keep the `DATA` folder where it is.

Experiment with different values and see how they change the simulated storm!

## Files in this Package

* `main.py`: The core Python code for the simulation. (You don't need to edit this unless you're an advanced user wanting to change the model itself).
* `parameters.txt`: **Your input settings for the simulation. This is the main file you will edit.**
* `requirements.txt`: Lists the Python libraries needed by the script (handled automatically by the run scripts).
* `run_simulation.sh`: Script to run the simulation on macOS/Linux.
* `run_simulation.bat`: Script to run the simulation on Windows.
* `README.md`: This instruction file.
* `DATA/` (folder): Contains map data.
    * `TOWN_MOI_1131028.shp` (and related files like .dbf, .shx): Shapefile for drawing Taiwan town boundaries.

Have fun exploring how weather works!