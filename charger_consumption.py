import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image

# Load the suburb coordinates
coordinates_df = pd.read_csv('suburb_locations.csv')
coordinates_df['coordinates'] = coordinates_df['coordinates'].apply(lambda x: eval(x))  # Convert string to tuple

# Load the optimization results
results_df = pd.read_csv('ev_charger_optimization_results.csv')

# Load the processed map image
map_image_path = 'images/suburbs_processed.png'
map_image = Image.open(map_image_path)

# Prepare data for power consumption
years = sorted(results_df['Year'].unique())

# Create folder to save frames
output_frames_path = 'power_consumption_frames'
if not os.path.exists(output_frames_path):
    os.makedirs(output_frames_path)

# Create a list to store frames
frames = []

# Generate frames for each year
for year in years:
    plt.figure(figsize=(12, 8))
    plt.imshow(map_image)  # Display the base map

    # Get the data for the current year
    year_data = results_df[results_df['Year'] == year].copy()

    # Merge with coordinates
    year_data = year_data.merge(coordinates_df, left_on='District', right_on='suburb', how='left')

    # Get the maximum power consumption for normalization
    max_power_consumption = year_data['Power Consumption from Chargers (kW)'].max()

    # Plot power consumption with normalized colors
    for i, row in year_data.iterrows():
        coords = row['coordinates']
        power_consumption = row['Power Consumption from Chargers (kW)']

        # Normalize the color intensity based on the power consumption
        normalized_intensity = power_consumption / max_power_consumption if max_power_consumption > 0 else 0
        color = plt.cm.Reds(normalized_intensity)  # Use a colormap (Reds)

        # Plot the suburb location with size proportional to power consumption
        plt.scatter(coords[0], coords[1], s=power_consumption * 10, alpha=0.7, color=color, edgecolor='none')

    plt.title(f'Power Consumption by Year {year}')
    plt.axis('off')  # Hide axes

    # Create the legend text as a running total for each suburb
    running_total_legend = "\n".join(
        [f"{row['District']} : {int(row['Power Consumption from Chargers (kW)'])}" for i, row in year_data.iterrows()]
    )

    # Add the running total legend in the bottom right corner
    plt.text(
        x=map_image.size[0] - 600,  # Position near the right edge
        y=map_image.size[1] - 100,  # Position near the bottom edge
        s=running_total_legend,
        fontsize=10,
        color='white',
        verticalalignment='bottom',
        horizontalalignment='left',
        bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3')  # Background box for better visibility
    )

    # Save the current frame
    frame_filename = os.path.join(output_frames_path, f'frame_{year}.png')
    plt.savefig(frame_filename, bbox_inches='tight', dpi=300)
    plt.close()

    # Append the frame to the list of frames
    frames.append(Image.open(frame_filename))

# Create a GIF from the frames
output_gif_path = 'power_consumption_growth.gif'
frames[0].save(output_gif_path, save_all=True, append_images=frames[1:], duration=500, loop=0)

print(f"Power consumption GIF saved to {output_gif_path}")
