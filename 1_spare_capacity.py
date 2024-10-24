import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
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

# Create a folder to save frames
output_frames_path = 'spare_capacity_frames'
if not os.path.exists(output_frames_path):
    os.makedirs(output_frames_path)

# Create a list to store frames for the GIF
frames = []


# To maintain consistent color scaling across all frames, find the overall min and max spare capacities
overall_min_capacity = results_df['Spare Capacity (MW)'].min()
overall_max_capacity = results_df['Spare Capacity (MW)'].max()

# Define normalization for a diverging colormap (to handle positive and negative values)
# We'll use a symmetric range around zero
abs_max = max(abs(overall_min_capacity), abs(overall_max_capacity))
norm = plt.Normalize(vmin=-abs_max, vmax=abs_max)

# Define the colormap: use a diverging colormap like 'bwr' or 'seismic'
cmap = plt.cm.bwr  # Blue-White-Red colormap for negative-positive differentiation

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

    # Separate positive and negative spare capacities
    positive_data = season_data[season_data['Spare Capacity (MW)'] >= 0]
    negative_data = season_data[season_data['Spare Capacity (MW)'] < 0]

    # Plot positive spare capacity
    if not positive_data.empty:
        colors_positive = cmap(norm(positive_data['Spare Capacity (MW)']))
        sizes_positive = positive_data['Spare Capacity (MW)'] * size_scale

        # Plot scatter for positive spare capacity
        plt.scatter(
            [coord[0] for coord in positive_data['coordinates']],
            [coord[1] for coord in positive_data['coordinates']],
            s=sizes_positive,
            alpha=0.7,
            color=colors_positive,
            edgecolor='k',
            linewidth=0.5,
            label='Positive Spare Capacity'
        )

    # Plot negative spare capacity
    if not negative_data.empty:
        # Use the same colormap for consistency
        colors_negative = cmap(norm(negative_data['Spare Capacity (MW)']))
        sizes_negative = negative_data['Spare Capacity (MW)'].abs() * size_scale  # Use absolute values for sizing

        plt.scatter(
            [coord[0] for coord in negative_data['coordinates']],
            [coord[1] for coord in negative_data['coordinates']],
            s=sizes_negative,
            alpha=0.7,
            color=colors_negative,
            edgecolor='k',
            linewidth=0.5,
            label='Negative Spare Capacity'
        )

    # Extract Season Type and Year for the title
    # Assuming results_df has 'Season Type' and 'Year' columns
    season_info = season_data[['Season', 'Year', 'Season Type']].drop_duplicates().iloc[0]
    season_number = season_info['Season']
    year = season_info['Year']
    season_type = season_info['Season Type']

    # Format the year with leading zero if necessary
    year_display = f"{int(year):02}" if int(year) < 10 else int(year)

    # Set the plot title
    plt.title(f'Spare Capacity in the Grid -\nYear {year_display}, {season_type}', fontsize=16)

    plt.axis('off')  # Hide axes for better visualization

    # Create the running total legend as a string
    running_total_legend = "\n".join(
        [f"{row['District']} : {int(row['Spare Capacity (MW)'])} MW" for _, row in season_data.iterrows()]
    )

    # Retrieve Overall Spare Capacity for the current season
    overall_spare_capacity = results_df[results_df['Season'] == season]['Overall Spare Capacity (MW)'].values
    if len(overall_spare_capacity) > 0:
        overall_spare_capacity = overall_spare_capacity[0]
        running_total_legend += f"\nOverall Spare Capacity: {int(overall_spare_capacity)} MW"
    else:
        overall_spare_capacity = None  # Handle missing column
        running_total_legend += f"\nOverall Spare Capacity: N/A"

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

    if not positive_data.empty:
        # Positive Spare Capacity handle
        positive_handle = Line2D([], [], marker='o', color='w',
                                  markerfacecolor=cmap(norm(positive_data['Spare Capacity (MW)'].max())),
                                  markersize=10, markeredgecolor='k',
                                  label='Positive Spare Capacity')
        handles.append(positive_handle)
        labels.append('Positive Spare Capacity')

    if not negative_data.empty:
        # Negative Spare Capacity handle
        negative_handle = Line2D([], [], marker='o', color='w',
                                  markerfacecolor=cmap(norm(negative_data['Spare Capacity (MW)'].min())),
                                  markersize=10, markeredgecolor='k',
                                  label='Negative Spare Capacity')
        handles.append(negative_handle)
        labels.append('Negative Spare Capacity')

    # Overall Spare Capacity handle (using a star marker)
    overall_handle = Line2D([], [], marker='*', color='gold', linestyle='None', markersize=15,
                            markeredgecolor='k', label='Overall Spare Capacity')
    handles.append(overall_handle)
    labels.append('Overall Spare Capacity')

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

# Ensure that there are frames to save
if frames:
    # Define the output GIF path
    output_gif_path = 'spare_capacity_growth.gif'

    # Save the frames as a GIF
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=500,  # Duration between frames in milliseconds
        loop=0          # Loop indefinitely
    )

    print(f"Spare capacity GIF saved to {output_gif_path}")
else:
    print("No frames were created. GIF will not be generated.")
