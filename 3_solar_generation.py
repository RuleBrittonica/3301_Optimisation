import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import ast  # For safer evaluation of coordinate strings
from matplotlib.lines import Line2D  # For custom legend entries

# ---------------------------
# Configuration Parameters
# ---------------------------

# File paths
SUBURB_COORDINATES_CSV = 'suburb_locations.csv'
OPTIMIZATION_RESULTS_CSV = 'ev_charger_optimization_results.csv'
MAP_IMAGE_PATH = 'images/suburbs_processed.png'
OUTPUT_FRAMES_DIR = 'solar_generation_frames'
OUTPUT_GIF_PATH = 'solar_generation_growth.gif'

# Visualization parameters
CIRCLE_COLOR = 'yellow'
CIRCLE_ALPHA = 0.7
CIRCLE_EDGE_COLOR = 'black'
CIRCLE_LINEWIDTH = 0.5
SIZE_SCALE = 1200
FONT_SIZE_TITLE = 16
FONT_SIZE_LEGEND = 10
FONT_SIZE_TEXT = 10

# ---------------------------
# Data Loading and Preparation
# ---------------------------

# Load the suburb coordinates
coordinates_df = pd.read_csv(SUBURB_COORDINATES_CSV)
# Safely convert string to tuple using ast.literal_eval
coordinates_df['coordinates'] = coordinates_df['coordinates'].apply(lambda x: ast.literal_eval(x))

# Load the optimization results
results_df = pd.read_csv(OPTIMIZATION_RESULTS_CSV)

print("Optimization Results Sample:")
print(results_df.head())
print("\nSuburb Coordinates Sample:")
print(coordinates_df.head())

# Load the processed map image
if not os.path.exists(MAP_IMAGE_PATH):
    raise FileNotFoundError(f"Map image not found at path: {MAP_IMAGE_PATH}")
map_image = Image.open(MAP_IMAGE_PATH)

# Prepare data for solar generation
years = sorted(results_df['Year'].unique())

# Create folder to save frames
if not os.path.exists(OUTPUT_FRAMES_DIR):
    os.makedirs(OUTPUT_FRAMES_DIR)

# Create a list to store frames
frames = []

# Define normalization based on solar generation values for consistent sizing across frames
solar_capacities = results_df['Solar Generation (MW)']
if not solar_capacities.empty:
    min_solar = solar_capacities.min()
    max_solar = solar_capacities.max()
    norm_solar = plt.Normalize(vmin=min_solar, vmax=max_solar)
else:
    # Default normalization if there are no solar capacities
    norm_solar = plt.Normalize(vmin=0, vmax=1)

# ---------------------------
# Visualization Loop
# ---------------------------

for year in years:
    plt.figure(figsize=(12, 8))
    plt.imshow(map_image)  # Display the base map

    # Get the data for the current year
    year_data = results_df[results_df['Year'] == year].copy()

    # Merge with coordinates
    year_data = year_data.merge(coordinates_df, left_on='District', right_on='suburb', how='left')

    # Handle missing coordinates
    missing_coords = year_data[year_data['coordinates'].isnull()]
    if not missing_coords.empty:
        print(f"Warning: Missing coordinates for the following districts in year {year}:")
        print(missing_coords['District'].tolist())
        # Drop these entries
        year_data = year_data.dropna(subset=['coordinates'])

    if year_data.empty:
        print(f"No data to plot for year {year}. Skipping this frame.")
        plt.close()
        continue  # Skip to the next year

    # Extract solar generation data
    solar_generation = year_data['Solar Generation (MW)']
    coordinates = year_data['coordinates'].tolist()

    # Calculate sizes based on solar generation
    sizes = solar_generation * SIZE_SCALE / max_solar if max_solar != 0 else solar_generation * SIZE_SCALE

    # Plot solar generation as yellow circles
    plt.scatter(
        [coord[0] for coord in coordinates],
        [coord[1] for coord in coordinates],
        s=sizes,
        alpha=CIRCLE_ALPHA,
        color=CIRCLE_COLOR,
        edgecolor=CIRCLE_EDGE_COLOR,
        linewidth=CIRCLE_LINEWIDTH,
        label='Solar Generation'
    )

    # Title
    display_year = f"{year:02}" if year < 10 else f"{year}"
    plt.title(f'Solar Generation in the Grid - Year {display_year}', fontsize=FONT_SIZE_TITLE)
    plt.axis('off')  # Hide axes

    # Create the running total legend as a string
    running_total_legend = "\n".join(
        [f"{row['District']} : {row['Solar Generation (MW)']:.2f} MW" for _, row in year_data.iterrows()]
    )
    # Calculate Overall Solar Generation for the current year
    overall_solar = solar_generation.sum()
    running_total_legend += f"\nOverall Solar Generation: {overall_solar:.2f} MW"

    # Add the running total legend in the bottom right corner
    plt.text(
        x=map_image.size[0] - 1200,
        y=map_image.size[1] - 100,
        s=running_total_legend,
        fontsize=FONT_SIZE_TEXT,
        color='white',
        verticalalignment='bottom',
        horizontalalignment='left',
        bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3')  # Background box for better visibility
    )

    # Create custom legend entries
    handles = []
    labels = []

    # Solar Generation handle
    solar_handle = Line2D([], [], marker='o', color='w', markerfacecolor=CIRCLE_COLOR, markersize=10,
                          markeredgecolor=CIRCLE_EDGE_COLOR, label='Solar Generation')
    handles.append(solar_handle)
    labels.append('Solar Generation')

    # Add the legend to the plot
    # plt.legend(handles=handles, labels=labels, loc='upper right', fontsize=FONT_SIZE_LEGEND, framealpha=0.6)

    # Save the current frame
    frame_filename = os.path.join(OUTPUT_FRAMES_DIR, f'frame_{year}.png')
    plt.savefig(frame_filename, bbox_inches='tight', dpi=300)
    plt.close()

    # Append the frame to the list of frames
    try:
        frames.append(Image.open(frame_filename))
        print(f"Processed year {year}")
    except Exception as e:
        print(f"Error loading frame for year {year}: {e}")

# ---------------------------
# Create GIF from Frames
# ---------------------------

# Ensure that there are frames to save
if frames:
    # Create a GIF from the frames
    frames[0].save(
        OUTPUT_GIF_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=500,  # Duration between frames in milliseconds
        loop=0
    )

    print(f"GIF saved to {OUTPUT_GIF_PATH}")
else:
    print("No frames were created. GIF will not be generated.")
