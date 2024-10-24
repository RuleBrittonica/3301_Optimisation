import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import ast  #For safer evaluation of coordinate strings
from matplotlib.lines import Line2D  #For custom legend entries

#---------------------------
#Configuration Parameters
#---------------------------

#File paths
SUBURB_COORDINATES_CSV = 'suburb_locations.csv'
OPTIMIZATION_RESULTS_CSV = 'ev_charger_optimization_results.csv'
MAP_IMAGE_PATH = 'images/suburbs_processed.png'
OUTPUT_FRAMES_DIR = 'solar_generation_frames'
OUTPUT_GIF_PATH = 'solar_generation_growth.gif'

#Visualization parameters
CIRCLE_COLOR = 'yellow'
CIRCLE_ALPHA = 0.7
CIRCLE_EDGE_COLOR = 'black'
CIRCLE_LINEWIDTH = 0.5
SIZE_SCALE = 1200
FONT_SIZE_TITLE = 16
FONT_SIZE_LEGEND = 10
FONT_SIZE_TEXT = 10

#---------------------------
#Data Loading and Preparation
#---------------------------

#Load the suburb coordinates
coordinates_df = pd.read_csv(SUBURB_COORDINATES_CSV)
#Safely convert string to tuple using ast.literal_eval
coordinates_df['coordinates'] = coordinates_df['coordinates'].apply(lambda x: ast.literal_eval(x))

#Load the optimization results
results_df = pd.read_csv(OPTIMIZATION_RESULTS_CSV)

print("Optimization Results Sample:")
print(results_df.head())
print("\nSuburb Coordinates Sample:")
print(coordinates_df.head())

#Load the processed map image
if not os.path.exists(MAP_IMAGE_PATH):
    raise FileNotFoundError(f"Map image not found at path: {MAP_IMAGE_PATH}")
map_image = Image.open(MAP_IMAGE_PATH)

#----------------------------------------
#2. Prepare for Plotting
#----------------------------------------

#Get all unique seasons sorted in order
seasons = sorted(results_df['Season'].unique())

#Create folder to save frames
if not os.path.exists(OUTPUT_FRAMES_DIR):
    os.makedirs(OUTPUT_FRAMES_DIR)

#Create a list to store frames
frames = []

#----------------------------------------
#3. Normalize Solar Generation Across All Seasons
#----------------------------------------

#To maintain consistent size scaling across all frames, find the overall maximum solar generation
overall_max_solar = results_df['Solar Generation (MW)'].max()

#Define normalization based on overall maximum solar generation
if overall_max_solar > 0:
    norm_solar = plt.Normalize(vmin=0, vmax=overall_max_solar)
else:
    norm_solar = plt.Normalize(vmin=0, vmax=1)  #Avoid division by zero

#----------------------------------------
#4. Generate Frames for Each Season
#----------------------------------------

for season in seasons:
    plt.figure(figsize=(12, 8))
    plt.imshow(map_image)  #Display the base map

    #Get the data for the current season
    season_data = results_df[results_df['Season'] == season].copy()

    #Merge with coordinates to get (x, y) positions
    season_data = season_data.merge(coordinates_df, left_on='District', right_on='suburb', how='left')

    #Handle missing coordinates
    missing_coords = season_data[season_data['coordinates'].isnull()]
    if not missing_coords.empty:
        print(f"Warning: Missing coordinates for the following districts in Season {season}:")
        print(missing_coords['District'].tolist())
        #Drop these entries
        season_data = season_data.dropna(subset=['coordinates'])

    if season_data.empty:
        print(f"No data to plot for Season {season}. Skipping this frame.")
        plt.close()
        continue  #Skip to the next season

    #Extract solar generation data
    solar_generation = season_data['Solar Generation (MW)']
    coordinates = season_data['coordinates'].tolist()

    #Calculate sizes based on solar generation
    sizes = solar_generation * SIZE_SCALE / overall_max_solar if overall_max_solar != 0 else solar_generation * SIZE_SCALE

    #Plot solar generation as yellow circles
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

    #Extract Season Type and Year for the title
    #Assuming 'Season Type' and 'Year' columns exist
    season_info = season_data[['Season', 'Year', 'Season Type']].drop_duplicates().iloc[0]
    season_number = season_info['Season']
    year = season_info['Year']
    season_type = season_info['Season Type']

    #Format the year with leading zero if necessary
    year_display = f"{int(year):02}" if int(year) < 10 else int(year)

    #Set the plot title
    plt.title(f'Solar Generation in the Grid -\nYear {year_display}, {season_type}', fontsize=FONT_SIZE_TITLE)

    plt.axis('off')  #Hide axes

    #Create the running total legend as a string
    running_total_legend = "\n".join(
        [f"{row['District']} : {row['Solar Generation (MW)']:.2f} MW" for _, row in season_data.iterrows()]
    )
    #Calculate Overall Solar Generation for the current season
    overall_solar = solar_generation.sum()
    running_total_legend += f"\nOverall Solar Generation: {overall_solar:.2f} MW"

    #Add the running total legend in the bottom left corner
    plt.text(
        x=50,  #Position near the left edge
        y=map_image.size[1] - 100,  #Position near the bottom edge
        s=running_total_legend,
        fontsize=FONT_SIZE_TEXT,
        color='white',
        verticalalignment='bottom',
        horizontalalignment='left',
        bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3')  #Background box for better visibility
    )

    #Create custom legend entries
    handles = []
    labels = []

    #Solar Generation handle
    solar_handle = Line2D([], [], marker='o', color='w', markerfacecolor=CIRCLE_COLOR, markersize=10,
                          markeredgecolor=CIRCLE_EDGE_COLOR, label='Solar Generation')
    handles.append(solar_handle)
    labels.append('Solar Generation')

    #Add the legend to the plot
    #plt.legend(handles=handles, labels=labels, loc='upper right', fontsize=FONT_SIZE_LEGEND, framealpha=0.6)

    #Save the current frame
    frame_filename = os.path.join(OUTPUT_FRAMES_DIR, f'frame_{season:02d}.png')  #Zero-padded season number
    plt.savefig(frame_filename, bbox_inches='tight', dpi=600)
    plt.close()

    #Append the frame to the list of frames
    try:
        frames.append(Image.open(frame_filename))
        print(f"Processed Season {season} (Year {year}, {season_type})")
    except Exception as e:
        print(f"Error loading frame for Season {season}: {e}")

#----------------------------------------
#5. Create the GIF
#----------------------------------------

#Ensure that there are frames to save
if frames:
    #Create a GIF from the frames
    frames[0].save(
        OUTPUT_GIF_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=500,  #Duration between frames in milliseconds
        loop=0          #Loop indefinitely
    )

    print(f"GIF saved to {OUTPUT_GIF_PATH}")
else:
    print("No frames were created. GIF will not be generated.")
