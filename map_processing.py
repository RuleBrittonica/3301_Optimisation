from PIL import Image
import numpy as np

# Load the image
image_path = 'images/suburbs.png'
image = Image.open(image_path).convert('RGBA')

# Convert image to numpy array
data = np.array(image)

# Create a mask for each color
black_border_mask = (data[:, :, 0] == 0) & (data[:, :, 1] == 0) & (data[:, :, 2] == 0)  # black border
white_background_mask = (data[:, :, 0] == 255) & (data[:, :, 1] == 255) & (data[:, :, 2] == 255)  # white background
colored_regions_mask = ~black_border_mask & ~white_background_mask  # all non-black and non-white areas
white_text_mask = (data[:, :, 0] == 255) & (data[:, :, 1] == 255) & (data[:, :, 2] == 255)  # white text

# Create new output array
output_data = np.ones_like(data) * 255  # Start with a white background

# Set black borders
output_data[black_border_mask] = [0, 0, 0, 255]  # Black

# Set colored regions to white
output_data[colored_regions_mask] = [255, 255, 255, 255]  # White

# Change white text to black
output_data[white_text_mask] = [0, 0, 0, 255]  # Black

# Convert the processed array back to an image
output_image = Image.fromarray(output_data, 'RGBA')

# Save the output image
output_path = 'images/suburbs_processed.png'
output_image.save(output_path)

output_path
