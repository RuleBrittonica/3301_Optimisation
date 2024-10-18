import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import ast  # For safer evaluation of coordinate strings
from matplotlib.lines import Line2D  # For custom legend entries

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

# Prepare data for spare capacity
years = sorted(results_df['Year'].unique())

# Create folder to save frames
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

# Generate frames for each year
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
        print(f"Warning: Missing coordinates for the following suburbs in year {year}:")
        print(missing_coords['District'].tolist())
        # Drop these entries
        year_data = year_data.dropna(subset=['coordinates'])

    if year_data.empty:
        print(f"No data to plot for year {year}. Skipping this frame.")
        plt.close()
        continue  # Skip to the next year

    # Separate positive and negative spare capacities
    positive_data = year_data[year_data['Spare Capacity (MW)'] >= 0]
    negative_data = year_data[year_data['Spare Capacity (MW)'] < 0]

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

    # If the year is a single digit, add a leading zero to make it better in the gif
    display_year = f"{year:02}" if year < 10 else f"{year}"
    plt.title(f'Spare Capacity in the Grid - Year {display_year}', fontsize=16)
    plt.axis('off')  # Hide axes

    # Create the running total legend as a string
    running_total_legend = "\n".join(
        [f"{row['District']} : {int(row['Spare Capacity (MW)'])} MW" for _, row in year_data.iterrows()]
    )
    # Retrieve Overall Spare Capacity for the current year
    overall_spare_capacity = results_df[results_df['Year'] == year]['Overall Spare Capacity (MW)'].values[0]
    running_total_legend += f"\nOverall Spare Capacity: {int(overall_spare_capacity)} MW"

    # Add the running total legend in the bottom right corner
    plt.text(
        x=map_image.size[0] - 1100,
        y=map_image.size[1] - 100,
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
        positive_handle = Line2D([], [], marker='o', color='w', markerfacecolor=cmap(0.5), markersize=10,
                                  markeredgecolor='k', label='Positive Spare Capacity')
        handles.append(positive_handle)
        labels.append('Positive Spare Capacity')

    if not negative_data.empty:
        # Negative Spare Capacity handle
        negative_handle = Line2D([], [], marker='o', color='w', markerfacecolor='red', markersize=10,
                                  markeredgecolor='k', label='Negative Spare Capacity')
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
    frame_filename = os.path.join(output_frames_path, f'frame_{year}.png')
    plt.savefig(frame_filename, bbox_inches='tight', dpi=300)
    plt.close()

    # Append the frame to the list of frames
    try:
        frames.append(Image.open(frame_filename))
        print(f"Processed year {year}")
    except Exception as e:
        print(f"Error loading frame for year {year}: {e}")

# Ensure that there are frames to save
if frames:
    # Create a GIF from the frames
    output_gif_path = 'spare_capacity_growth.gif'
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=500,  # Duration between frames in milliseconds
        loop=0
    )

    print(f"GIF saved to {output_gif_path}")
else:
    print("No frames were created. GIF will not be generated.")
