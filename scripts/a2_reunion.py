#!/usr/bin/python3
# -*- coding: utf-8 -*
# 影像分块后，将分块影像拼接成一张影像

import os
import re
from PIL import Image

def reunite_blocks(input_folder, output_image_path, blocks_per_row):
    """
    Reunites blocks into a single image based on naming sequence.

    :param input_folder: Folder where the blocks are stored.
    :param output_image_path: Path to save the reunited image.
    :param blocks_per_row: Number of blocks in each row.
    """
    # List all the block image files and sort them by the numerical part of the filename
    block_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.png')]
    block_files.sort(key=lambda x: int(re.search(r'(\d+)\.png$', x).group(1)))

    # Load the first block to get the block size
    first_block = Image.open(block_files[0])
    block_width, block_height = first_block.size

    # Determine the total number of rows (assuming all rows are filled)
    total_rows = len(block_files) // blocks_per_row

    # Create a new image with the total size
    total_width = block_width * blocks_per_row
    total_height = block_height * total_rows
    reunited_image = Image.new('RGBA', (total_width, total_height))

    # Paste the blocks into the image
    for index, block_file in enumerate(block_files):
        block = Image.open(block_file)
        # Calculate the position of the current block
        x_index = index % blocks_per_row
        y_index = index // blocks_per_row
        x = x_index * block_width
        y = (total_rows - 1 - y_index) * block_height  # Start from the bottom
        reunited_image.paste(block, (x, y))

    # Save the reunited image
    reunited_image.save(output_image_path)

# User parameters
input_folder = 'G:/Project/'   # TODO: image path
output_image_path = 'G:/B_Project/'   # TODO: image path
blocks_per_row = 13  # TODO: This should be set by the number of blocks from a same row

# Reunite the blocks into a single image
reunite_blocks(input_folder, output_image_path, blocks_per_row)
