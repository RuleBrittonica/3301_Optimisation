import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import ast  # For safer evaluation of coordinate strings
from matplotlib.lines import Line2D  # For custom legend entries

# ----------------------------------------
# 1. Load Necessary Data
# ----------------------------------------

# Load the suburb coordinates
coordinates_df = pd.read_csv('suburb_locations.csv')
# Safely convert string to tuple using ast.literal_eval
coordinates_df['coordinates'] = coordinates_df['coordinates'].apply(lambda x: ast.literal_eval(x))

# Load the optimization results
results_df = pd.read_csv('ev_charger_optimization_results.csv')

print("Optimization Results Sample:")
print(results_df.head())
print("\nSuburb Coordinates Sample:")
print(coordinates_df.head())

# Load the processed map image
map_image_path = '../images/suburbs_processed.png'
map_image = Image.open(map_image_path)

# ----------------------------------------
# 2. Prepare for Plotting
# ----------------------------------------

# Get all unique seasons sorted in order
seasons = sorted(results_df['Season'].unique())

# Create a folder to save frames
output_frames_path = 'total_charger_frames'
if not os.path.exists(output_frames_path):
    os.makedirs(output_frames_path)

# Create a list to store frames
frames = []

# Define the colormap for chargers
cmap = plt.cm.Oranges  # Using Oranges colormap

# Precompute overall maximum chargers for normalization
overall_max_chargers = results_df['Total Chargers Placed'].max()
if overall_max_chargers > 0:
    norm = plt.Normalize(vmin=0, vmax=overall_max_chargers)
else:
    norm = plt.Normalize(vmin=0, vmax=1)  # Avoid division by zero

# Scaling factor for circle sizes to make them visually distinguishable
size_scale = 15

# ----------------------------------------
# 3. Generate Frames for Each Season
# ----------------------------------------

for season in seasons:
    plt.figure(figsize=(12, 8))
    plt.imshow(map_image)  # Display the base map

    # Get the data for the current season
    season_data = results_df[results_df['Season'] == season].copy()

    # Merge with coordinates to get (x, y) positions
    season_data = season_data.merge(coordinates_df, left_on='District', right_on='suburb', how='left')

    # Handle missing coordinates
    missing_coords = season_data[season_data['coordinates'].isnull()]
    if not missing_coords.empty:
        print(f"Warning: Missing coordinates for the following suburbs in Season {season}:")
        print(missing_coords['District'].tolist())
        # Drop these entries
        season_data = season_data.dropna(subset=['coordinates'])

    if season_data.empty:
        print(f"No data to plot for Season {season}. Skipping this frame.")
        plt.close()
        continue  # Skip to the next season

    # Plot chargers cumulatively with normalized colors
    for i, row in season_data.iterrows():
        coords = row['coordinates']
        cumulative_chargers = row['Total Chargers Placed']

        # Normalize the color intensity based on the number of chargers
        normalized_intensity = cumulative_chargers / overall_max_chargers if overall_max_chargers > 0 else 0
        color = cmap(norm(cumulative_chargers))  # Use the normalized value

        # Plot the suburb location with size proportional to cumulative chargers
        plt.scatter(
            coords[0],
            coords[1],
            s=cumulative_chargers * size_scale,
            alpha=0.7,
            color=color,
            edgecolor='none'
        )

    # Extract Season Type and Year for the title
    # Assuming results_df has 'Season Type' and 'Year' columns
    if 'Season Type' in season_data.columns and 'Year' in season_data.columns:
        season_info = season_data[['Season', 'Year', 'Season Type']].drop_duplicates().iloc[0]
        season_number = season_info['Season']
        year = season_info['Year']
        season_type = season_info['Season Type']
    else:
        # Fallback if columns are missing
        season_number = season
        year = 'Unknown'
        season_type = 'Unknown'

    # Format the year with leading zero if necessary
    if isinstance(year, (int, float)) and year < 10:
        year_display = f'0{int(year)}'
    else:
        year_display = int(year) if isinstance(year, (int, float)) else str(year)

    # Set the plot title
    plt.title(f'Total EV Chargers Added - \n Year {year_display}, Season {season_type}', fontsize=12)
    plt.axis('off')  # Hide axes

    # Create the legend text as a running total for each suburb
    running_total_legend = "\n".join(
        [f"{row['District']} : {int(row['Total Chargers Placed'])}" for i, row in season_data.iterrows()]
    )

    # Retrieve Overall Spare Capacity for the current season
    # if 'Overall Spare Capacity (MW)' in season_data.columns:
    #     overall_spare_capacity = season_data['Overall Spare Capacity (MW)'].values[0]
    #     running_total_legend += f"\nOverall Spare Capacity: {int(overall_spare_capacity)} MW"
    # else:
    #     overall_spare_capacity = None  # Handle missing column
    #     running_total_legend += "\nOverall Spare Capacity: N/A"

    # Add the running total legend in the bottom right corner
    plt.text(
        x=50,  # Position near the right edge
        y=map_image.size[1] - 100,   # Position near the bottom edge
        s=running_total_legend,
        fontsize=10,
        color='white',
        verticalalignment='bottom',
        horizontalalignment='left',
        bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3')  # Background box for better visibility
    )

    # Save the current frame
    frame_filename = os.path.join(output_frames_path, f'frame_{season:02d}.png')  # Zero-padded season number
    plt.savefig(frame_filename, bbox_inches='tight', dpi=600)
    plt.close()

    # Append the frame to the list of frames
    try:
        frames.append(Image.open(frame_filename))
        print(f"Processed Season {season} (Year {year_display}, {season_type})")
    except Exception as e:
        print(f"Error loading frame for Season {season}: {e}")

# ----------------------------------------
# 4. Create the GIF
# ----------------------------------------

# Ensure that there are frames to save
if frames:
    # Define the output GIF path
    output_gif_path = 'chargers_growth.gif'

    # Save the frames as a GIF
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=500,  # Duration between frames in milliseconds
        loop=0          # Loop indefinitely
    )

    print(f"GIF saved to {output_gif_path}")
else:
    print("No frames were created. GIF will not be generated.")
