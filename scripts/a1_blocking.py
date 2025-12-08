#!/usr/bin/python3
# -*- coding: utf-8 -*
# 影像分块
"""
Notice:
# For 0.5 * 0.5 meter resolution, we should set the block size larger, approximately as 1024 * 1024 pixels,
# For 0.75 * 0.75 meter resolution, we should set the block size smaller, approximately as 640 * 640 pixels
"""

import os
from PIL import Image, ImageFile
import numpy as np
# Increase the maximum size
Image.MAX_IMAGE_PIXELS = 1000000000
ImageFile.LOAD_TRUNCATED_IMAGES = True

def split_image_into_blocks(image, block_size, output_folder, base_filename):
    """
    Splits the image into blocks of the given size starting from the bottom left corner,
    going right and then upwards, and saves them to the output folder with the base filename
    followed by an index.
    """
    # Create the output directory if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get the image dimensions
    width, height = image.size

    # Calculate the number of blocks in each dimension
    x_blocks = width // block_size
    y_blocks = height // block_size

    # Split the image into blocks and save each block
    block_index = 0
    for y_block in range(y_blocks - 1, -1, -1):  # Start from the bottom row and go upwards
        for x_block in range(x_blocks):  # Go from left to right
            # Calculate the region of the current block
            left = x_block * block_size
            upper = y_block * block_size
            right = left + block_size
            lower = upper + block_size

            # Extract and save the block
            block = image.crop((left, upper, right, lower))
            block_filename = f"{base_filename}__{block_index}.png"
            block_path = os.path.join(output_folder, block_filename)
            block.save(block_path)
            block_index += 1


# Load the image
input_image_path = 'G:/B_Project/B_FCs/a_workspace/a_oridata/JL1_Chongming_resize/JL1_Chongming1_20221002.png'  # TODO: image path
image = Image.open(input_image_path)

# Ensure the image is in RGBA format
if image.mode != 'RGBA':
    image = image.convert('RGBA')

# Find the bounding rectangle of the key area
data = np.array(image)
alpha_channel = data[:, :, 3]
non_empty_columns = np.where(alpha_channel.max(axis=0) > 0)[0]
non_empty_rows = np.where(alpha_channel.max(axis=1) > 0)[0]
bounding_box = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))

# Set the block size
block_size = 640  # TODO: block size

# Pad the image so that the bounding box dimensions are multiples of block_size
row_min, row_max, col_min, col_max = bounding_box
row_padding = (block_size - ((row_max - row_min) % block_size)) % block_size
col_padding = (block_size - ((col_max - col_min) % block_size)) % block_size

# Pad the rows and columns equally on both sides
top_padding = row_padding // 2
bottom_padding = row_padding - top_padding
left_padding = col_padding // 2
right_padding = col_padding - left_padding

# Pad the bounding box with zeros (for the alpha channel as well)
padded_image = np.pad(data[row_min:row_max+1, col_min:col_max+1, :],
                      ((top_padding, bottom_padding), (left_padding, right_padding), (0, 0)),
                      mode='constant', constant_values=0)

# Convert the padded image back to an Image object
padded_image = Image.fromarray(padded_image)

# Define the output folder and base filename
output_folder = 'G:/B_Project/'   # TODO: image path
base_filename = ''  # TODO: image base name

# Split the image into blocks and save them
split_image_into_blocks(padded_image, block_size, output_folder, base_filename)
