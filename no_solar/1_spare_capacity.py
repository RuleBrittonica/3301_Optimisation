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
output_frames_path = 'spare_capacity_frames'
if not os.path.exists(output_frames_path):
    os.makedirs(output_frames_path)

# Create a list to store frames
frames = []

# Define the colormap for positive spare capacity
cmap = plt.cm.Purples  # Using Purples colormap for a single color gradient

# Define normalization based on positive spare capacity values for consistent coloring across frames
positive_capacities = results_df[results_df['Spare Capacity (MW)'] >= 0]['Spare Capacity (MW)']
if not positive_capacities.empty:
    min_positive_capacity = positive_capacities.min()
    max_positive_capacity = positive_capacities.max()
    norm_positive = plt.Normalize(vmin=min_positive_capacity, vmax=max_positive_capacity)
else:
    # Default normalization if there are no positive capacities
    norm_positive = plt.Normalize(vmin=0, vmax=1)

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

    # Separate positive and negative spare capacities
    positive_data = season_data[season_data['Spare Capacity (MW)'] >= 0]
    negative_data = season_data[season_data['Spare Capacity (MW)'] < 0]

    # Plot positive spare capacity
    if not positive_data.empty:
        colors_positive = cmap(norm_positive(positive_data['Spare Capacity (MW)']))
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
        # Assign bright red color for negative spare capacity
        colors_negative = ['red'] * len(negative_data)
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
    season_info = season_data[['Season', 'Year', 'Season Type']].drop_duplicates().iloc[0]
    season_number = season_info['Season']
    year = season_info['Year']
    season_type = season_info['Season Type']

    # Format the year with leading zero if necessary
    year_display = f'0{int(year)}' if year < 10 else int(year)

    # Set the plot title
    plt.title(f'Spare Capacity in the Grid - \n Year {year_display}, Season {season_type}', fontsize=12)
    plt.axis('off')  # Hide axes

    # Create the running total legend as a string
    running_total_legend = "\n".join(
        [f"{row['District']} : {int(row['Spare Capacity (MW)'])} MW"
         for _, row in season_data.iterrows()]
    )
    # Retrieve Overall Spare Capacity for the current season
    overall_spare_capacity = season_data['Overall Spare Capacity (MW)'].values[0] + 20
    running_total_legend += f"\nOverall Spare Capacity: {int(overall_spare_capacity)} MW"

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
        print(f"Processed Season {season} (Year {year}, {season_type})")
    except Exception as e:
        print(f"Error loading frame for Season {season}: {e}")

# ----------------------------------------
# 4. Create the GIF
# ----------------------------------------

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

    print(f"GIF saved to {output_gif_path}")
else:
    print("No frames were created. GIF will not be generated.")
