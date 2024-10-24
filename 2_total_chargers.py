import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import os
import ast
from matplotlib.lines import Line2D  


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
map_image_path = 'images/suburbs_processed.png'
map_image = Image.open(map_image_path)

# Get all unique seasons sorted in order
seasons = sorted(results_df['Season'].unique())

# Create folder to save frames
output_frames_path = 'total_charger_frames'
if not os.path.exists(output_frames_path):
    os.makedirs(output_frames_path)

# Create a list to store frames
frames = []

# To maintain consistent color scaling across all frames, find the overall maximum chargers placed
overall_max_chargers = results_df['Total Chargers Placed'].max()

# Define normalization based on overall maximum chargers placed
if overall_max_chargers > 0:
    norm = plt.Normalize(vmin=0, vmax=overall_max_chargers)
else:
    norm = plt.Normalize(vmin=0, vmax=1)  # Avoid division by zero

# Define the colormap
cmap = plt.cm.Oranges  # Using Oranges colormap for chargers

# Scaling factor for circle sizes to make them visually distinguishable
size_scale = 15

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
        print(f"Warning: Missing coordinates for the following districts in Season {season}:")
        print(missing_coords['District'].tolist())
        # Drop these entries
        season_data = season_data.dropna(subset=['coordinates'])

    if season_data.empty:
        print(f"No data to plot for Season {season}. Skipping this frame.")
        plt.close()
        continue  # Skip to the next season

    # Plot chargers cumulatively with normalized colors
    colors = cmap(norm(season_data['Total Chargers Placed']))
    sizes = season_data['Total Chargers Placed'] * size_scale

    plt.scatter(
        [coord[0] for coord in season_data['coordinates']],
        [coord[1] for coord in season_data['coordinates']],
        s=sizes,
        alpha=0.7,
        color=colors,
        edgecolor='none'
    )

    # Extract Season Type and Year for the title
    # Assuming 'Season Type' and 'Year' columns exist
    season_info = season_data[['Season', 'Year', 'Season Type']].drop_duplicates().iloc[0]
    season_number = season_info['Season']
    year = season_info['Year']
    season_type = season_info['Season Type']

    # Format the year with leading zero if necessary
    year_display = f"{int(year):02}" if int(year) < 10 else int(year)

    # Set the plot title
    plt.title(f'Total EV Chargers Added -\nYear {year_display}, {season_type}', fontsize=16)

    plt.axis('off')  # Hide axes

    # Create the running total legend as a string
    running_total_legend = "\n".join(
        [f"{row['District']} : {int(row['Total Chargers Placed'])}" for _, row in season_data.iterrows()]
    )

    # Add the running total legend in the bottom left corner
    plt.text(
        x=50,  # Position near the left edge
        y=map_image.size[1] - 100,  # Position near the bottom edge
        s=running_total_legend,
        fontsize=10,
        color='white',
        verticalalignment='bottom',
        horizontalalignment='left',
        bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3')  # Background box for better visibility
    )

    # Create custom legend entries
    handles = []
    labels = []

    # Positive Spare Capacity handle
    positive_handle = Line2D([], [], marker='o', color='w',
                              markerfacecolor=cmap(norm(overall_max_chargers)),
                              markersize=10, markeredgecolor='k',
                              label='Total Chargers Placed')
    handles.append(positive_handle)
    labels.append('Total Chargers Placed')

    # Add the legend to the plot
    # plt.legend(handles=handles, labels=labels, loc='upper right', fontsize=10, framealpha=0.6)

    # Save the current frame
    frame_filename = os.path.join(output_frames_path, f'frame_{season:02d}.png')  # Zero-padded season number
    plt.savefig(frame_filename, bbox_inches='tight', dpi=600)
    plt.close()

    # Append the frame to the list of frames
    try:
        frames.append(Image.open(frame_filename))
        print(f"Processed Season {season} (Year {year}, {season_type})")
    except Exception as e:
        print(f"Error loading frame for Season {season}: {e}")

# ----------------------------------------
# 5. Create the GIF
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
