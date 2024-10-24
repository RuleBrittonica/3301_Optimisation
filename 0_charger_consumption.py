import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
import ast



# Load the suburb coordinates
coordinates_df = pd.read_csv('suburb_locations.csv')
# Safely convert string to tuple using ast.literal_eval
coordinates_df['coordinates'] = coordinates_df['coordinates'].apply(lambda x: ast.literal_eval(x))

# Load the optimization results
results_df = pd.read_csv('ev_charger_optimization_results.csv')

# Load the processed map image
map_image_path = 'images/suburbs_processed.png'  # Adjusted path as per your script
map_image = Image.open(map_image_path)

# Get all unique seasons sorted in order
seasons = sorted(results_df['Season'].unique())

# Create a folder to save frames
output_frames_path = 'power_consumption_frames'
if not os.path.exists(output_frames_path):
    os.makedirs(output_frames_path)

# Create a list to store frames for the GIF
frames = []

# To maintain consistent color scaling across all frames, find the overall maximum power consumption
overall_max_power_consumption = results_df['Power Consumption from Chargers (MW)'].max()
if overall_max_power_consumption > 0:
    norm = plt.Normalize(vmin=0, vmax=overall_max_power_consumption)
else:
    norm = plt.Normalize(vmin=0, vmax=1)  # Avoid division by zero

# Define the colormap
cmap = plt.cm.Reds  # Using Reds colormap

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

    # Plot power consumption for each district
    for _, row in season_data.iterrows():
        coords = row['coordinates']
        power_consumption = row['Power Consumption from Chargers (MW)']

        # Normalize the color intensity based on the power consumption
        normalized_intensity = norm(power_consumption)
        color = cmap(normalized_intensity)  # Use the Reds colormap with normalized intensity

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
    plt.title(f'Charger Power Consumption -\nYear {year_display}, {season_type}', fontsize=16)

    plt.axis('off')  # Hide axes for better visualization

    # Create the legend text as a running total for each district
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
    try:
        frames.append(Image.open(frame_filename))
        print(f"Processed Season {season} (Year {year}, {season_type})")
    except Exception as e:
        print(f"Error loading frame for Season {season}: {e}")


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
        loop=0          # Loop indefinitely
    )

    print(f"Power consumption GIF saved to {output_gif_path}")
else:
    print("No frames were created. GIF will not be generated.")
