# Taipei Afternoon Thunderstorm Simulation

A Python-based meteorological simulation tool for modeling afternoon thunderstorm development over Taipei, Taiwan. This educational simulation demonstrates how various atmospheric parameters interact to produce localized convective precipitation events typical of summer afternoons in the Taipei basin.

## Overview

This simulation models the development of afternoon thunderstorms from initial morning conditions (09:00 LST) to peak afternoon convection (14:00 LST). It generates precipitation and cloud field visualizations based on user-defined atmospheric parameters, with a focus on whether conditions produce significant rainfall at National Taiwan University (NTU).

### Key Features
- Interactive parameter configuration via `parameters.txt`
- Realistic precipitation pattern generation based on atmospheric instability
- Wind field visualization with convergence effects
- Cloud coverage modeling
- Success/failure threshold based on NTU precipitation (>25 mm/hr)
- Optional town boundary overlay using Taiwan government shapefiles

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Quick Start

#### Windows
1. Clone or download this repository
2. Navigate to the `simulation` folder
3. Edit `parameters.txt` with your desired values
4. Double-click `run_simulation.bat`

#### macOS/Linux
1. Clone or download this repository
2. Navigate to the `simulation` folder
3. Make the script executable: `chmod +x run_simulation.sh`
4. Edit `parameters.txt` with your desired values
5. Run: `./run_simulation.sh`

### Manual Installation
If you prefer to set up manually:

```bash
cd simulation
pip install -r requirements.txt
python main.py
```

### Dependencies
- **Required**: `numpy`, `matplotlib`, `tqdm`
- **Optional**: `geopandas` (for town boundary visualization)

**Note**: If you encounter issues installing `geopandas`, the simulation will still run without map boundaries. To install geopandas dependencies:
- **Ubuntu/Debian**: `sudo apt-get install libgdal-dev gdal-bin python3-gdal`
- **macOS**: `brew install gdal geos proj`
- **Windows**: Consider using conda instead of pip

## Configuration

Edit `parameters.txt` before running the simulation. All parameters represent initial conditions at 09:00 LST in Taipei:

```ini
[DEFAULT]
# Solar Radiation: From 1 to 10 (Integer ONLY)
solar_radiation = 9

# Cloud Coverage: From 0.0 to 1.0
initial_cloud_coverage = 0.15

# Average Wind Speed (m/s): From 0.0 to 10.0 
average_wind_speed = 5.0

# Wind Direction Grid: 3x3 grid of wind directions (degrees FROM North)
# Format: [[NW,N,NE],[W,C,E],[SW,S,SE]]
wind_direction_grid = [[240,235,230],[235,225,220],[230,220,215]]

# Relative Humidity: From 0 to 100 (Integer ONLY)
relative_humidity = 90

# Near Surface Temperature (Celsius): From 25.0 to 30.0
near_surface_temp_c = 30.0

# Air Temperature at 5km height (Celsius): From -5.0 to -15.0
upper_air_temp_c = -15.0

# Optional: Path to Taiwan town boundaries shapefile (leave unchanged if not available)
# town_shapefile_path = ./DATA/TOWN_MOI_1131028.shp
```

### Parameter Guidelines

**For successful thunderstorm generation (>25 mm/hr at NTU), try:**
- High solar radiation (8-10)
- Low initial cloud coverage (<0.3)
- Moderate wind speed (3-6 m/s)
- Southwest wind direction (220-240°)
- High humidity (>80%)
- Large temperature difference (>40°C between surface and upper air)

## Usage

1. **Configure parameters**: Edit `parameters.txt` with your desired atmospheric conditions
2. **Run simulation**: Execute the appropriate run script for your OS
3. **View results**: Check the generated PNG files:
   - `tpe_precip_1400.png` - Precipitation field at 14:00 LST
   - `tpe_cloud_1400.png` - Cloud coverage and wind field at 14:00 LST

### Understanding the Output

The simulation will display:
- Initial parameters summary
- Progress bar during computation
- Final results showing SUCCESS or FAILURE based on the 25 mm/hr threshold at NTU
- Location of saved visualization files

### Precipitation Color Scale
- White: <2.5 mm/hr (no significant precipitation)
- Blue: 2.5-15 mm/hr (light rain)
- Green: 15-30 mm/hr (moderate rain)
- Yellow/Orange: 30-50 mm/hr (heavy rain)
- Red: 50-80 mm/hr (very heavy rain)
- Purple: >80 mm/hr (extreme rain)

## Scientific Background

The simulation models several key atmospheric processes:

1. **Atmospheric Instability**: Calculated from the temperature difference between surface and upper air
2. **Convection Potential**: Combines solar radiation, moisture, wind, and instability
3. **Storm Development**: Uses Gaussian distributions with wind advection
4. **Terrain Effects**: Enhanced precipitation when southwest winds interact with Taipei basin topography

## Troubleshooting

### Common Issues

1. **"parameters.txt not found"**: Ensure you're running from the `simulation` directory
2. **Import errors**: Run the installation commands or use the provided scripts
3. **No map boundaries**: This is normal if geopandas isn't installed - the simulation still works
4. **Low precipitation values**: Try adjusting parameters according to the guidelines above

### Getting Help

If you encounter issues:
1. Check that all files are in the correct locations
2. Verify Python 3.7+ is installed: `python --version`
3. Try the manual installation steps
4. For geopandas issues, the simulation works without it

## Optional: Taiwan Shapefile Data

To display town boundaries, download the official Taiwan township boundary shapefile:
1. Visit the Taiwan Government Open Data Platform
2. Search for "TOWN_MOI" shapefile data
3. Place the `.shp` file (and associated files) in `simulation/DATA/`
4. Update the path in `parameters.txt` if needed

## Educational Notes

This simulation is designed for educational purposes to demonstrate:
- How multiple atmospheric variables interact to produce thunderstorms
- The importance of atmospheric instability in convection
- Why afternoon thunderstorms are common in tropical/subtropical regions
- The role of local topography in precipitation enhancement

## Technical Details

- **Grid Resolution**: 0.01° (~1.1 km)
- **Domain**: 121.35°E-121.75°E, 24.85°N-25.25°N (covering greater Taipei)
- **Time**: Simulates conditions at 14:00 LST based on 09:00 LST initialization
- **Random Seed**: Fixed at 38 for reproducible results

## Author

**Author**: Yu-Chen Chen (Cubee)
**Contact**: cubee0405@gmail.com
**Institution**: Department of Atmospheric Sciences at National Taiwan University
**Date**: 2025

## License

This project is provided for educational and research purposes. Please cite the author if using this code in academic work.

## Acknowledgments

- Taiwan Central Weather Bureau for meteorological insights
- National Taiwan University for location data
- Taiwan Government for open shapefile data

---

*Note: This is a conceptual model for educational purposes and should not be used for operational weather forecasting.*