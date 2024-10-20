import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image

# ----------------------------------------
# 1. Load Necessary Data
# ----------------------------------------

# Load the suburb coordinates
coordinates_df = pd.read_csv('suburb_locations.csv')
coordinates_df['coordinates'] = coordinates_df['coordinates'].apply(lambda x: eval(x))  # Convert string to tuple

# Load the optimization results
results_df = pd.read_csv('ev_charger_optimization_results.csv')

# Load the processed map image
map_image_path = '../images/suburbs_processed.png'
map_image = Image.open(map_image_path)

# ----------------------------------------
# 2. Prepare for Plotting
# ----------------------------------------

# Get all unique seasons sorted in order
seasons = sorted(results_df['Season'].unique())

# Create a folder to save frames
output_frames_path = 'power_consumption_frames'
if not os.path.exists(output_frames_path):
    os.makedirs(output_frames_path)

# Create a list to store frames for the GIF
frames = []

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

    # Get the maximum power consumption for normalization (for color intensity)
    max_power_consumption = season_data['Power Consumption from Chargers (MW)'].max()

    # Plot power consumption for each district
    for _, row in season_data.iterrows():
        coords = row['coordinates']
        power_consumption = row['Power Consumption from Chargers (MW)']

        # Normalize the color intensity based on the power consumption
        normalized_intensity = power_consumption / max_power_consumption if max_power_consumption > 0 else 0
        color = plt.cm.Reds(normalized_intensity)  # Use the Reds colormap

        # Plot the suburb location with size proportional to power consumption
        plt.scatter(
            coords[0],
            coords[1],
            s=power_consumption * 15,  # Scale size for visibility
            alpha=0.7,
            color=color,
            edgecolor='none'
        )

    # Extract Season Type and Year for the title
    season_info = season_data[['Season', 'Year', 'Season Type']].drop_duplicates().iloc[0]
    season_number = season_info['Season']
    year = season_info['Year']
    season_type = season_info['Season Type']

    # Format the year with leading zero if necessary
    year_display = f'0{int(year)}' if year < 10 else int(year)

    # Set the plot title
    plt.title(f'Charger Power Consumption - \n Year {year_display}, Season {season_type}', fontsize=12)

    plt.axis('off')  # Hide axes for better visualization

    # Create the legend text as a running total for each suburb
    running_total_legend = "\n".join(
        [f"{row['District']} : {int(row['Power Consumption from Chargers (MW)'])} MW"
         for _, row in season_data.iterrows()]
    )

    # Add the running total legend in the bottom right corner
    plt.text(
        x=50,  # Position near the right edge
        y=map_image.size[1] - 100,  # Position near the bottom edge
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
    frames.append(Image.open(frame_filename))

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
    output_gif_path = 'power_consumption_growth.gif'

    # Save the frames as a GIF
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=500,  # Duration between frames in milliseconds
        loop=0  # Loop indefinitely
    )

    print(f"GIF saved to {output_gif_path}")
else:
    print("No frames were created. GIF will not be generated.")
