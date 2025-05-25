import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
import random
import warnings
import os
import ast
import time
import configparser

from tqdm import tqdm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("Warning: geopandas library not found. Boundary plotting will be skipped.")
    print("If you want to see map boundaries, try to install it (see README).")

warnings.filterwarnings(
    "ignore",
    message=".*tight_layout.*",
    category=UserWarning
)

# --- Model Configuration Constants (can be overridden by parameters.txt for path) ---
MIN_LON, MAX_LON = 121.35, 121.75
MIN_LAT, MAX_LAT = 24.85, 25.25
GRID_RESOLUTION = 0.01 # Grid resolution in degrees

NTU_LON, NTU_LAT = 121.5345, 25.0170
NTU_PRECIP_THRESHOLD = 25.0
CONFIG_SEED = 38

# This will be read from parameters.txt and can be overridden there
TOWN_SHAPEFILE_PATH_CONFIG = "./DATA/TOWN_MOI_1131028.shp"

# --- Global dictionary for parameters ---
params_global = {}


def nonlinspace(start, end, intervals, bounds):
    N    = len(intervals)
    if N != len(bounds) + 1:
        raise ValueError("len(intervals) should be len(bounds) + 1")

    tmp = np.concatenate(([start], bounds, [end]))
    segments = [np.arange(tmp[i], tmp[i+1], intervals[i]) for i in range(N-1)]
    segments.append(np.arange(tmp[-2], tmp[-1]+intervals[-1]/2, intervals[-1]))
    return np.concatenate(segments)

def calculate_atmospheric_instability_index(surface_temp_c, upper_air_temp_c):
    temp_diff = surface_temp_c - upper_air_temp_c
    min_diff_for_instability = 20.0 # Min temp diff (C) for any notable instability
    max_diff_for_high_instability = 45.0 # Temp diff (C) for highest instability index

    if temp_diff <= min_diff_for_instability:
        instability_index = 1.0 # Low instability
    elif temp_diff >= max_diff_for_high_instability:
        instability_index = 10.0 # High instability
    else:
        # Linear interpolation for instability index between 1.0 and 10.0
        instability_index = 1.0 + 9.0 * \
            (temp_diff - min_diff_for_instability) / \
            (max_diff_for_high_instability - min_diff_for_instability)
    return float(instability_index)


def calculate_convection_potential(current_params):
    potential = 0.0
    # Solar radiation effect
    potential += current_params['solar_radiation'] * 1.8
    # Initial cloud coverage effect
    if current_params['initial_cloud_coverage'] < 0.1: # Very clear
        potential += 3.0
    elif current_params['initial_cloud_coverage'] < 0.35: # Partially clear
        potential += (0.35 - current_params['initial_cloud_coverage']) * 15 # Bonus for less cloud
    else: # Cloudier
        potential -= (current_params['initial_cloud_coverage'] - 0.35) * 20 # Penalty for more cloud
    # Wind speed effect
    if 1.0 <= current_params['average_wind_speed'] <= 6.0: # Optimal wind range
        potential += current_params['average_wind_speed'] * 1.0
    elif current_params['average_wind_speed'] < 1.0: # Too calm
        potential -= 2.0
    else: # Too windy
        potential -= (current_params['average_wind_speed'] - 6.0) * 1.5
    # Moisture availability effect
    potential += current_params['moisture_availability'] * 2.2
    # Atmospheric instability effect
    potential += current_params['atmospheric_instability_index'] * 2.5
    # Terrain and wind direction interaction (conceptual SW flow interaction with Taipei basin terrain)
    avg_wind_deg = np.mean(current_params['wind_direction_grid'])
    # Influence factor peaks when wind is from SW (225 degrees)
    sw_influence_factor = max(0, np.cos(np.deg2rad(avg_wind_deg - 225)))
    terrain_forcing_base = 5.0 # Base terrain effect
    terrain_forcing_sw_bonus = 2.0 * sw_influence_factor # Bonus for SW flow
    TERRAIN_FORCING = terrain_forcing_base + terrain_forcing_sw_bonus
    potential += TERRAIN_FORCING

    return max(0, potential) # Potential cannot be negative

def get_wind_from_grid(lon, lat):
    # Access params_global directly as it's populated in main()
    lon_edges = np.linspace(MIN_LON, MAX_LON, 4) # 3x3 grid means 4 edges
    lat_edges = np.linspace(MIN_LAT, MAX_LAT, 4)

    # Find which grid cell (lon, lat) falls into
    grid_j = np.searchsorted(lon_edges, lon) - 1
    grid_i = np.searchsorted(lat_edges, lat) - 1

    # Clip to ensure indices are within 0-2 range for a 3x3 grid
    grid_j = np.clip(grid_j, 0, 2)
    grid_i = np.clip(grid_i, 0, 2) # Latitude index is often inverted in grids (0 is top)

    # Wind direction from the grid (Note: y-axis on plots is lat, often maps to rows, hence 2-grid_i for typical array indexing)
    direction = params_global['wind_direction_grid'][2 - grid_i, grid_j] # 2-grid_i if lat_grid[0] is MIN_LAT
    speed = params_global['average_wind_speed'] # Use the single average speed for this conceptual model
    return direction, speed

def generate_precipitation_field(convection_potential_precip):
    # Access params_global directly
    lon_grid = np.arange(MIN_LON, MAX_LON + GRID_RESOLUTION/2, GRID_RESOLUTION) # Ensure MAX_LON is included
    lat_grid = np.arange(MIN_LAT, MAX_LAT + GRID_RESOLUTION/2, GRID_RESOLUTION) # Ensure MAX_LAT is included
    combined_precipitation = np.zeros((len(lat_grid), len(lon_grid)))

    # Initial storm center (randomized within a region)
    initial_storm_lon = 121.50 + random.uniform(-0.06, 0.06)
    initial_storm_lat = 25.05 + random.uniform(-0.05, 0.05)
    initial_storm_lon = np.clip(initial_storm_lon, MIN_LON + 0.05, MAX_LON - 0.05) # Keep within bounds
    initial_storm_lat = np.clip(initial_storm_lat, MIN_LAT + 0.05, MAX_LAT - 0.05)

    # Advection of the storm center based on wind at its initial location
    center_wind_dir, center_wind_speed = get_wind_from_grid(initial_storm_lon, initial_storm_lat)
    wind_rad_movement = np.deg2rad(center_wind_dir + 180) # Storm moves WITH the wind, source of wind + 180 for vector direction
                                                        # Or, if wind_dir is where wind is GOING TO, then just center_wind_dir.
                                                        # Meteorologically, wind direction is FROM where it blows. So if wind is 225 (SW), it blows TO NE.
                                                        # For particle movement, if wind is FROM SW, U is +ve, V is +ve.
                                                        # Angle for movement vector: 270 - meteorological_direction.
                                                        # Let's assume wind direction is meteorological (FROM where it blows).
                                                        # Movement angle in radians = np.deg2rad(270 - center_wind_dir)
    movement_angle_rad = np.deg2rad(270 - center_wind_dir) # Standard conversion from meteorological to Cartesian angle

    # Conceptual displacement factor - reduced to make storms more localized if desired
    displacement_factor = 0.004 * (center_wind_speed / 5.0) # Scaled by wind speed

    main_advected_lon = initial_storm_lon + displacement_factor * np.cos(movement_angle_rad)
    main_advected_lat = initial_storm_lat + displacement_factor * np.sin(movement_angle_rad)
    main_advected_lon = np.clip(main_advected_lon, MIN_LON + 0.05, MAX_LON - 0.05)
    main_advected_lat = np.clip(main_advected_lat, MIN_LAT + 0.05, MAX_LAT - 0.05)

    final_storm_center_for_report = (main_advected_lon, main_advected_lat)

    CONVECTION_THRESHOLD_FOR_RAIN = 16.0 # Min convection potential to generate rain
    INTENSITY_SCALER = 0.9 # Scales convection potential to precipitation intensity
    MAX_RANDOM_BOOST_TO_INTENSITY = 2.8 # Max random addition to intensity

    overall_base_precip_intensity = 0.0
    if convection_potential_precip > CONVECTION_THRESHOLD_FOR_RAIN:
        overall_base_precip_intensity = (convection_potential_precip - CONVECTION_THRESHOLD_FOR_RAIN) * INTENSITY_SCALER
        overall_base_precip_intensity += random.uniform(0, MAX_RANDOM_BOOST_TO_INTENSITY)
        overall_base_precip_intensity = max(0, overall_base_precip_intensity)

    if overall_base_precip_intensity > 1.0: # Only proceed if there's some intensity
        num_cores = 1 # Default to one precipitation core
        # Determine number of cores based on intensity and instability
        if overall_base_precip_intensity > 15.0 and params_global['atmospheric_instability_index'] > 7:
            num_cores = random.randint(2, 3)
        elif overall_base_precip_intensity > 8.0 and params_global['atmospheric_instability_index'] > 5:
            num_cores = random.randint(1, 2)
        else: # For very high intensity or moderate intensity with some instability, allow for multiple cores via weighted choice
            if overall_base_precip_intensity > 20:
                num_cores = random.choices([1,2,3], weights=[0.6, 0.3, 0.1])[0]
            elif overall_base_precip_intensity > 10:
                 num_cores = random.choices([1,2], weights=[0.7,0.3])[0]
            else:
                num_cores = 1


        for core_idx in range(num_cores):
            core_precipitation = np.zeros((len(lat_grid), len(lon_grid)))
            instability_offset_scale = params_global['atmospheric_instability_index'] / 10.0

            if num_cores > 1:
                # Offset secondary cores from the main advected center
                offset_lon = random.uniform(-0.05, 0.05) * (1 + instability_offset_scale) # Larger offset with more instability
                offset_lat = random.uniform(-0.05, 0.05) * (1 + instability_offset_scale)
                core_center_lon = main_advected_lon + offset_lon
                core_center_lat = main_advected_lat + offset_lat
                if core_idx == 0: # The first core can also be the 'main' reported center if it's multi-core
                    final_storm_center_for_report = (core_center_lon, core_center_lat)
            else: # Single core situation
                core_center_lon = main_advected_lon
                core_center_lat = main_advected_lat
                final_storm_center_for_report = (core_center_lon, core_center_lat) # This is the only center

            core_center_lon = np.clip(core_center_lon, MIN_LON + 0.05, MAX_LON - 0.05)
            core_center_lat = np.clip(core_center_lat, MIN_LAT + 0.05, MAX_LAT - 0.05)

            core_intensity = overall_base_precip_intensity / num_cores # Distribute intensity among cores
            if num_cores > 1 :
                core_intensity *= random.uniform(0.6, 1.4) # Vary intensity of individual cores

            # Elongation and orientation based on local wind at the core's center
            core_wind_dir_deg, _ = get_wind_from_grid(core_center_lon, core_center_lat)
            # Elongation typically along the wind vector (or perpendicular for squall lines, here simpler)
            # Let's make elongation angle related to 90 degrees from wind direction (cross-wind)
            base_elongation_angle_rad_for_core = np.deg2rad( (270 - core_wind_dir_deg) + 90 ) # Perpendicular to wind vector

            # Core-specific properties for multi-core storms
            if num_cores > 1 and core_idx > 0: # For secondary cores
                random_orientation_offset_deg = random.uniform(-60, 60)
                current_elongation_angle_rad = base_elongation_angle_rad_for_core + np.deg2rad(random_orientation_offset_deg)
                current_elongation_strength = random.uniform(1.2, 2.2) # More variable elongation
                current_rand_strength_mult = random.uniform(0.6, 1.4) # More variable size/strength
            else: # For the first/only core
                current_elongation_angle_rad = base_elongation_angle_rad_for_core
                current_elongation_strength = 1.6 # Default elongation
                current_rand_strength_mult = random.uniform(0.9, 1.1) # Tighter variation

            # Sigma (spread) values for Gaussian cores
            spread_factor_base_val_precip = 0.030 # Base spread in degrees
            spread_factor_instability_bonus_precip = (params_global['atmospheric_instability_index'] / 10.0) * 0.020 # Instability increases spread
            base_s_sigma_precip = spread_factor_base_val_precip + spread_factor_instability_bonus_precip

            sigma_major = base_s_sigma_precip * current_elongation_strength * current_rand_strength_mult
            sigma_minor = base_s_sigma_precip / current_elongation_strength * current_rand_strength_mult
            sigma_major = max(sigma_major, 0.005) # Prevent zero or too small sigma
            sigma_minor = max(sigma_minor, 0.005)

            # Final small random perturbation to the angle
            angle_perturb_rad = np.deg2rad(random.uniform(-25, 25))
            final_angle_rad = current_elongation_angle_rad + angle_perturb_rad

            cos_a = np.cos(final_angle_rad)
            sin_a = np.sin(final_angle_rad)

            # Coordinate warping for irregularity (more pronounced for larger cores)
            warp_strength_val = random.uniform(0.003, 0.012) * (base_s_sigma_precip / 0.03)
            warp_freq_lon1 = random.uniform(15, 35) # Waviness frequency
            warp_freq_lat1 = random.uniform(15, 35)
            warp_freq_lon2 = random.uniform(20, 40) # Second harmonic for complexity
            warp_freq_lat2 = random.uniform(20, 40)
            warp_phase_lon1 = random.uniform(0, 2 * np.pi) # Random phases
            warp_phase_lat1 = random.uniform(0, 2 * np.pi)
            warp_phase_lon2 = random.uniform(0, 2 * np.pi)
            warp_phase_lat2 = random.uniform(0, 2 * np.pi)


            for i, lat_val in enumerate(lat_grid):
                for j, lon_val in enumerate(lon_grid):
                    dx_orig = lon_val - core_center_lon
                    dy_orig = lat_val - core_center_lat

                    # Apply coordinate warping
                    term1_dx = np.sin(dx_orig * warp_freq_lon1 + dy_orig * warp_freq_lat1 * 0.3 + warp_phase_lon1)
                    term2_dx = np.sin(dx_orig * warp_freq_lon2 * 0.6 + dy_orig * warp_freq_lat2 + warp_phase_lon2)
                    warp_dx = warp_strength_val * (term1_dx * 0.6 + term2_dx * 0.4)

                    term1_dy = np.cos(dy_orig * warp_freq_lat1 + dx_orig * warp_freq_lon1 * 0.3 + warp_phase_lat1)
                    term2_dy = np.cos(dy_orig * warp_freq_lat2 * 0.6 + dx_orig * warp_freq_lon2 + warp_phase_lat2)
                    warp_dy = warp_strength_val * (term1_dy * 0.6 + term2_dy * 0.4)

                    dx = dx_orig + warp_dx
                    dy = dy_orig + warp_dy

                    # Rotate coordinates for elongated Gaussian
                    dx_rot = dx * cos_a + dy * sin_a
                    dy_rot = -dx * sin_a + dy * cos_a

                    # Calculate Gaussian precipitation value
                    # Add 1e-7 to sigma to prevent division by zero if sigma is extremely small
                    dist_sq_mod = (dx_rot / (sigma_major + 1e-7))**2 + (dy_rot / (sigma_minor + 1e-7))**2
                    gaussian_precip = core_intensity * np.exp(-0.5 * dist_sq_mod)

                    precip_value = gaussian_precip
                    # Add spatial noise to make it less smooth, scaled by precip intensity
                    if gaussian_precip > 0.15: # Only add noise to significant precip
                        spatial_noise_factor = gaussian_precip * 0.15 # Noise is a percentage of intensity
                        precip_value += random.uniform(-0.5, 0.5) * spatial_noise_factor # Balanced noise
                    elif gaussian_precip < 0.02 : # Threshold out very light, almost zero, precipitation
                        precip_value = 0.0

                    core_precipitation[i, j] = max(0, precip_value) # Ensure non-negative
            combined_precipitation += core_precipitation # Add this core's rain to the total

    return lon_grid, lat_grid, combined_precipitation, final_storm_center_for_report


def generate_conceptual_cloud_field(lon_grid, lat_grid, convection_potential_cloud, final_storm_center_for_report):
    # Access params_global directly
    combined_cloud_field = np.zeros((len(lat_grid), len(lon_grid)))
    CLOUD_CONVECTION_THRESHOLD = 30.0 # Convection potential needed for significant cloud
    CLOUD_INTENSITY_SCALER = 0.050 # Scales potential to cloud density (0-1)

    overall_base_cloud_density = 0.0
    if convection_potential_cloud > CLOUD_CONVECTION_THRESHOLD:
        overall_base_cloud_density = min(1.0, (convection_potential_cloud - CLOUD_CONVECTION_THRESHOLD) * CLOUD_INTENSITY_SCALER)
        overall_base_cloud_density += random.uniform(0, 0.1) # Small random boost
        overall_base_cloud_density = min(1.0, max(0, overall_base_cloud_density))

    if overall_base_cloud_density > 0.05: # Generate distinct cloud cores if density is significant
        num_cloud_cores = 1
        if overall_base_cloud_density > 0.5 and params_global['atmospheric_instability_index'] > 6:
            num_cloud_cores = random.randint(2,3)
        elif overall_base_cloud_density > 0.2 and params_global['atmospheric_instability_index'] > 4:
            num_cloud_cores = random.randint(1,2)

        main_cloud_center_lon, main_cloud_center_lat = final_storm_center_for_report # Clouds centered around storm

        for _ in range(num_cloud_cores):
            core_cloud_field = np.zeros((len(lat_grid), len(lon_grid)))
            instability_c_offset_scale = params_global['atmospheric_instability_index'] / 10.0
            if num_cloud_cores > 1: # Offset secondary cloud cores
                offset_lon_c = random.uniform(-0.06, 0.06) * (1 + instability_c_offset_scale)
                offset_lat_c = random.uniform(-0.06, 0.06) * (1 + instability_c_offset_scale)
                core_c_lon = main_cloud_center_lon + offset_lon_c
                core_c_lat = main_cloud_center_lat + offset_lat_c
            else:
                core_c_lon = main_cloud_center_lon
                core_c_lat = main_cloud_center_lat

            core_c_lon = np.clip(core_c_lon, MIN_LON + 0.05, MAX_LON - 0.05)
            core_c_lat = np.clip(core_c_lat, MIN_LAT + 0.05, MAX_LAT - 0.05)

            core_cloud_density = overall_base_cloud_density / num_cloud_cores
            if num_cloud_cores > 1:
                core_cloud_density *= random.uniform(0.7, 1.3) # Vary density
            core_cloud_density = min(1.0, max(0, core_cloud_density))

            # Cloud cores are generally larger and more diffuse
            cloud_spread_factor_base_val = 0.040 # Larger base spread for clouds
            cloud_spread_factor_inst_bonus = (params_global['atmospheric_instability_index'] / 10.0) * 0.030
            base_s_sigma_cloud = cloud_spread_factor_base_val + cloud_spread_factor_inst_bonus

            spread_cx_mult = random.uniform(0.8, 1.2) # Random variation in spread
            spread_cy_mult = random.uniform(0.8, 1.2)

            sigma_cloud_lon = base_s_sigma_cloud * spread_cx_mult
            sigma_cloud_lat = base_s_sigma_cloud * spread_cy_mult
            sigma_cloud_lon = max(sigma_cloud_lon, 0.01) # Prevent too small sigma
            sigma_cloud_lat = max(sigma_cloud_lat, 0.01)

            for i, lat_val in enumerate(lat_grid):
                for j, lon_val in enumerate(lon_grid):
                    # Simple Gaussian for cloud cores (not elongated for simplicity)
                    dist_sq_c_mod = ((lon_val - core_c_lon)**2 / (sigma_cloud_lon**2 + 1e-7)) + \
                                    ((lat_val - core_c_lat)**2 / (sigma_cloud_lat**2 + 1e-7))
                    cloud_value = core_cloud_density * np.exp(-0.5 * dist_sq_c_mod)
                    # Add a small fraction of initial cloud coverage as background, plus noise
                    cloud_value += params_global['initial_cloud_coverage'] * 0.10 # Background from initial
                    cloud_value += random.uniform(-0.02, 0.02) # Slight general noise
                    core_cloud_field[i, j] = min(1.0, max(0, cloud_value)) # Clamp 0-1
            combined_cloud_field += core_cloud_field

        combined_cloud_field = np.clip(combined_cloud_field, 0, 1.0) # Ensure final field is 0-1
    else:
        # If not enough convection for distinct clouds, use a slightly modified initial coverage
        base_initial_cloud = params_global['initial_cloud_coverage'] * random.uniform(0.7,1.1)
        combined_cloud_field.fill(min(1.0, base_initial_cloud + random.uniform(0,0.03)))
        combined_cloud_field = np.clip(combined_cloud_field, 0, 1.0)

    return combined_cloud_field


def get_precipitation_at_ntu(lon_grid, lat_grid, precipitation):
    # Check if NTU coordinates are within the grid
    if NTU_LON < lon_grid[0] or NTU_LON > lon_grid[-1] or \
       NTU_LAT < lat_grid[0] or NTU_LAT > lat_grid[-1]:
        print(f"Warning: NTU coordinates ({NTU_LON}, {NTU_LAT}) are outside the simulation grid ({lon_grid[0]}-{lon_grid[-1]}, {lat_grid[0]}-{lat_grid[-1]}). Returning 0 precipitation.")
        return 0.0
    # Find the nearest grid point to NTU
    ntu_j_idx = np.argmin(np.abs(lon_grid - NTU_LON))
    ntu_i_idx = np.argmin(np.abs(lat_grid - NTU_LAT))
    return precipitation[ntu_i_idx, ntu_j_idx]

def plot_precipitation_styled(fig, ax, lon_grid, lat_grid, precipitation, ntu_precip_val):
    # Access params_global for TOWN_SHAPEFILE_PATH_CONFIG
    mpl.rcParams['legend.fontsize'] = 10
    ax.set_facecolor('whitesmoke') # Light gray background
    X, Y = np.meshgrid(lon_grid, lat_grid)

    # Define precipitation levels for color scale (non-linear)
    precip_levels = nonlinspace(2.5, 80, (2.5, 5, 10), (15, 50)) # Intervals, then split points

    # Custom colormap for precipitation (similar to weather radar)
    precip_colors_hex = ['#a0fffa','#00cdff','#0096ff', # Blues (light precip)
                         '#0069ff','#329600','#32ff00', # Greens (moderate)
                         '#ffff00','#ffc800','#ff9600', # Yellows/Oranges (heavy)
                         '#ff0000','#c80000','#a00000', # Reds (very heavy)
                         '#96009b','#c800d2','#ff00f5',] # Purples (extreme)
    cmap = mplc.ListedColormap(precip_colors_hex).with_extremes(under='#ffffff', over='#ffc8ff') # White for under, light purple for over
    norm = mplc.BoundaryNorm(precip_levels, cmap.N)

    # Plot precipitation data
    contourf_plot = ax.pcolormesh(X, Y, precipitation, cmap=cmap, norm=norm, zorder=2, shading='auto')

    # Plot town boundaries if geopandas is available and shapefile exists
    shapefile_to_use = params_global.get('town_shapefile_path', TOWN_SHAPEFILE_PATH_CONFIG) # Get from loaded params or default
    if GEOPANDAS_AVAILABLE and os.path.exists(shapefile_to_use):
        try:
            gdf_town = gpd.read_file(shapefile_to_use)
            gdf_town.plot(ax=ax, lw=0.5, fc='none', ec='dimgray', zorder=3) # Plot boundaries
        except Exception as e:
            print(f"Error plotting town boundaries from '{shapefile_to_use}': {e}")
    elif GEOPANDAS_AVAILABLE and not os.path.exists(shapefile_to_use):
        print(f"Warning: Shapefile not found at '{shapefile_to_use}'. Boundaries will not be plotted.")

    # Colorbar
    axins = inset_axes(ax, width="3%", height="100%", loc="center left",
                       bbox_to_anchor=(1.02, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0.3)
    cbar = fig.colorbar(contourf_plot, cax=axins, extend='both', ticks=precip_levels)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("mm/hr", fontsize=8)


    # Plot NTU station marker
    ax.plot(NTU_LON, NTU_LAT, 'ro', markersize=10 , mec='black', mew=1, label=f'NTU ({ntu_precip_val:.2f} mm/hr)', zorder=4)
    ax.legend(loc='upper right', facecolor='white', framealpha=0.7, fontsize=8)

    ax.set_xlabel('Longitude (°E)', fontsize=9)
    ax.set_ylabel('Latitude (°N)', fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_title("14:00 LST", fontsize=12, loc='right', fontweight='bold')
    ax.set_title(f"NTU Simulated Precipitation: {ntu_precip_val:.2f} mm/hr", fontsize=12, loc='left', fontweight='bold', color='darkred')
    ax.set_xlim(MIN_LON, MAX_LON)
    ax.set_ylim(MIN_LAT, MAX_LAT)
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for colorbar


def plot_cloud_and_wind(fig, ax, lon_grid, lat_grid, cloud_field, final_storm_center_for_report):
    # Access params_global for TOWN_SHAPEFILE_PATH_CONFIG and wind params
    ax.set_facecolor('lightcyan') # Sky-like background
    X, Y = np.meshgrid(lon_grid, lat_grid)

    # Cloud field plotting
    cloud_cmap = plt.colormaps.get_cmap("Greys")
    cloud_norm = mplc.BoundaryNorm(np.linspace(0.0, 1.0, 21), cloud_cmap.N, clip=True) # Smoother cloud gradient

    axins_cloud = inset_axes(ax, width="3%", height="100%", loc="center left",
                             bbox_to_anchor=(1.02, 0., 1., 1.), bbox_transform=ax.transAxes, borderpad=0.3)

    if np.any(cloud_field > 0.05): # Only plot if there are significant clouds
        contourf_cloud = ax.pcolormesh(X, Y, cloud_field, cmap=cloud_cmap, norm=cloud_norm, zorder=2, alpha=0.75, shading='auto')
        cbar_cloud = fig.colorbar(contourf_cloud, cax=axins_cloud, extend='neither', ticks=np.linspace(0.0, 1.0, 6))
        cbar_cloud.ax.tick_params(labelsize=8)
    else:
        axins_cloud.set_visible(False) # Hide colorbar if no significant cloud


    # Plot town boundaries
    shapefile_to_use = params_global.get('town_shapefile_path', TOWN_SHAPEFILE_PATH_CONFIG)
    if GEOPANDAS_AVAILABLE and os.path.exists(shapefile_to_use):
        try:
            gdf_town = gpd.read_file(shapefile_to_use)
            gdf_town.plot(ax=ax, lw=0.5, fc='none', ec='darkslategrey', zorder=3)
        except Exception as e:
            print(f"Error plotting town boundaries for cloud plot: {e}")

    # Wind field plotting (Quiver plot)
    skip = 3 # Plot every 'skip'th arrow for clarity
    X_quiver, Y_quiver = X[::skip, ::skip], Y[::skip, ::skip]
    u_plot = np.zeros_like(X_quiver)
    v_plot = np.zeros_like(Y_quiver)
    s_center_lon, s_center_lat = final_storm_center_for_report # Center for convergence effects

    for i_q in range(X_quiver.shape[0]):
        for j_q in range(X_quiver.shape[1]):
            plot_lon, plot_lat = X_quiver[i_q,j_q], Y_quiver[i_q,j_q]
            # Get regional wind based on the 3x3 grid
            regional_dir_deg, regional_speed = get_wind_from_grid(plot_lon, plot_lat)

            # Add minor random variation to regional wind for visual texture
            noisy_dir_deg = regional_dir_deg + random.uniform(-10, 10)
            noisy_speed = regional_speed * random.uniform(0.9, 1.1)
            noisy_speed = max(0.1, noisy_speed) # Ensure speed is not zero

            # Convert meteorological direction (FROM) to Cartesian angle (TO, counter-clockwise from East)
            regional_angle_rad = np.deg2rad(270 - noisy_dir_deg)
            u_regional_noisy = noisy_speed * np.cos(regional_angle_rad)
            v_regional_noisy = noisy_speed * np.sin(regional_angle_rad)

            # Conceptual convergence towards the main storm center
            dx_storm = s_center_lon - plot_lon
            dy_storm = s_center_lat - plot_lat
            dist_to_storm = np.sqrt(dx_storm**2 + dy_storm**2)

            convergence_strength = params_global['average_wind_speed'] * 0.6 * (1 + params_global['atmospheric_instability_index']/20.0)
            convergence_factor = convergence_strength * np.exp(-dist_to_storm * 25) # Exponential decay

            u_conv, v_conv = 0, 0
            if dist_to_storm > 0.001: # Avoid division by zero
                u_conv = convergence_factor * (dx_storm / dist_to_storm) # Towards storm center
                v_conv = convergence_factor * (dy_storm / dist_to_storm)

            final_u = u_regional_noisy + u_conv
            final_v = v_regional_noisy + v_conv
            u_plot[i_q,j_q] = final_u
            v_plot[i_q,j_q] = final_v

    Q = ax.quiver(X_quiver, Y_quiver, u_plot, v_plot, color='blue', scale=70,
                  width=0.0040, headwidth=4, zorder=4, alpha=0.65)
    ax.quiverkey(Q, X=0.925, Y=1.055, U=5, label='5 m/s', labelpos='E',
                 fontproperties={'size': 9}, color='blue')

    ax.plot(NTU_LON, NTU_LAT, 'ro', markersize=10 , mec='black', mew=1, label='NTU', zorder=4)
    ax.legend(loc='upper right', fontsize=8, facecolor='white', framealpha=0.7)
    ax.set_xlabel('Longitude (°E)', fontsize=9)
    ax.set_ylabel('Latitude (°N)', fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_title("NTU Simulated Wind field \nand Cloud concentration (0-1)", fontsize=12, loc='left', fontweight='bold', color='darkred')
    ax.set_title("14:00 LST", fontsize=12, loc='right', fontweight='bold')
    ax.set_xlim(MIN_LON, MAX_LON)
    ax.set_ylim(MIN_LAT, MAX_LAT)
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout


def load_parameters(filepath="parameters.txt"):
    global params_global # Ensure we're modifying the global dictionary
    global TOWN_SHAPEFILE_PATH_CONFIG # To update the default if specified in file

    config = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
    if not os.path.exists(filepath):
        print(f"ERROR: Parameter file '{filepath}' not found!")
        print("Please create it based on the example in README.md or the provided template.")
        exit(1)
    try:
        config.read(filepath)
        # Use 'DEFAULT' section as keys are not in a specific section in the example .txt
        cfg = config['DEFAULT']

        params_global['solar_radiation'] = cfg.getint('solar_radiation')
        params_global['initial_cloud_coverage'] = cfg.getfloat('initial_cloud_coverage')
        params_global['average_wind_speed'] = cfg.getfloat('average_wind_speed')

        wind_direction_str = cfg.get('wind_direction_grid')
        try:
            wind_direction_list = ast.literal_eval(wind_direction_str)
            wind_direction_matrix = np.array(wind_direction_list, dtype=float)
            if wind_direction_matrix.shape != (3,3):
                raise ValueError("Wind direction grid must be 3x3.")
            if not np.all((wind_direction_matrix >= 0) & (wind_direction_matrix <= 360)):
                raise ValueError("Wind directions must be between 0 and 360 degrees.")
            params_global['wind_direction_grid'] = wind_direction_matrix
        except Exception as e:
            print(f"Error parsing 'wind_direction_grid' from '{filepath}': {e}")
            print(f"Received: {wind_direction_str}")
            print("Ensure it is a valid Python list of lists format, e.g., [[225,225,225],[225,225,225],[225,225,225]]")
            exit(1)
        params_global['relative_humidity'] = cfg.getint('relative_humidity')
        params_global['moisture_availability'] = int(cfg.getint('relative_humidity')/10.0)
        params_global['near_surface_temp_c'] = cfg.getfloat('near_surface_temp_c')
        params_global['upper_air_temp_c'] = cfg.getfloat('upper_air_temp_c')

        # Update the shapefile path if specified, otherwise keep the default
        TOWN_SHAPEFILE_PATH_CONFIG = cfg.get('town_shapefile_path', fallback=TOWN_SHAPEFILE_PATH_CONFIG).strip()
        params_global['town_shapefile_path'] = TOWN_SHAPEFILE_PATH_CONFIG

    except Exception as e:
        print(f"Error reading or parsing parameters from '{filepath}': {e}")
        print("Please ensure the file is correctly formatted and all expected parameters are present.")
        exit(1)

    # --- Derived parameters ---
    params_global['atmospheric_instability_index'] = calculate_atmospheric_instability_index(
        params_global['near_surface_temp_c'], params_global['upper_air_temp_c']
    )
    return params_global


def validate_parameters():
    # Access params_global directly
    if not (1 <= params_global['solar_radiation'] <= 10):
        raise ValueError("Solar radiation (from parameters.txt) must be 1-10.")
    if not (0.0 <= params_global['initial_cloud_coverage'] <= 1.0):
        raise ValueError("Initial cloud coverage (from parameters.txt) must be 0.0-1.0.")
    if params_global['average_wind_speed'] < 0:
        raise ValueError("Average wind speed (from parameters.txt) cannot be negative.")
    if not (1 <= params_global['moisture_availability'] <= 10):
        raise ValueError("Relative humidity (from parameters.txt) must be 0-100.")

    temp_diff_check = params_global['near_surface_temp_c'] - params_global['upper_air_temp_c']
    if temp_diff_check <= 5: # Arbitrary small difference warning
        print(f"Warning: Low vertical temperature difference ({temp_diff_check:.1f}°C: {params_global['near_surface_temp_c']}°C surface, {params_global['upper_air_temp_c']}°C upper air). May result in low instability and minimal convection.")
    if temp_diff_check < 0:
        print(f"Warning: Negative vertical temperature difference (temperature inversion: {params_global['near_surface_temp_c']}°C surface, {params_global['upper_air_temp_c']}°C upper air). This typically suppresses convection.")


def main():
    global params_global # Ensure main uses the global dict
    np.random.seed(CONFIG_SEED)
    random.seed(CONFIG_SEED)

    print("===================== START OF SIMULATION =====================")
    print("Loading parameters...")
    load_parameters()
    validate_parameters()


    print("=========== INITIAL PARAMETERS at 09:00 LST @Taipei ===========")
    print(f"Solar radiation: {params_global['solar_radiation']}")
    print(f"Initial cloud coverage: {params_global['initial_cloud_coverage']:.2f}")
    print(f"Average wind speed: {params_global['average_wind_speed']:.1f} m/s")
    print(f"Wind direction grid (degrees from N): \n{params_global['wind_direction_grid']}")
    print(f"Relative humidity {params_global['relative_humidity']}%")
    print(f"Near surface temperature: {params_global['near_surface_temp_c']:.1f} C")
    print(f"Upper air temperature: {params_global['upper_air_temp_c']:.1f} C")
    print("=============================================================")
    
    print("Running Simulation...")
    for i in tqdm(range(10)):
        time.sleep(0.5)

    # --- Main ---
    convection_potential = calculate_convection_potential(params_global)
    lon_grid, lat_grid, precipitation_field, final_storm_center = generate_precipitation_field(convection_potential)
    ntu_precipitation = get_precipitation_at_ntu(lon_grid, lat_grid, precipitation_field)

    # --- Plotting ---
    fig_precip, ax_precip = plt.subplots(figsize=(8, 7))
    fig_precip.subplots_adjust(right=0.83)
    plot_precipitation_styled(fig_precip, ax_precip, lon_grid, lat_grid, precipitation_field, ntu_precipitation)
    precip_fname = "tpe_precip_1400.png"
    plt.savefig(precip_fname, dpi=300)
    plt.close(fig_precip)

    cloud_field = generate_conceptual_cloud_field(lon_grid, lat_grid, convection_potential, final_storm_center)
    fig_cloud, ax_cloud = plt.subplots(figsize=(8, 7))
    fig_cloud.subplots_adjust(right=0.79)

    plot_cloud_and_wind(fig_cloud, ax_cloud, lon_grid, lat_grid, cloud_field, final_storm_center)
    cloud_fname = "tpe_cloud_1400.png"
    plt.savefig(cloud_fname, dpi=300)
    plt.close(fig_cloud)

    print("\n========================== RESULTS ==========================")
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)

    if ntu_precipitation > NTU_PRECIP_THRESHOLD:
        print(f"SUCCESS! \nPrecipitation at NTU ({ntu_precipitation:.2f} mm hr⁻¹) is OVER the threshold of {NTU_PRECIP_THRESHOLD:.1f} mm hr⁻¹.")
    else:
        print(f"FAILED! \nPrecipitation at NTU ({ntu_precipitation:.2f} mm hr⁻¹) is BELOW the threshold of {NTU_PRECIP_THRESHOLD:.1f} mm hr⁻¹.")
    
    print(f"\n'{precip_fname}' AND '{cloud_fname}' HAS BEEN SAVED.")
    print(f"GO CHECK OUT IN THE FOLDER {current_dir}!")
    print("===================== END OF SIMULATION =====================")

if __name__ == "__main__":
    main()