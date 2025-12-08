# calculate the metrics/featrues of color, geometry, and distribution pattern for instance entities
import cv2
import numpy as np
import torch
import time
import pickle
import csv
import pandas as pd
import matplotlib.pyplot as plt
import re
import copy
from PIL import Image, ExifTags
from pathlib import Path
from scipy import ndimage
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree

input_folder = Path("G:/Project")  # TODO: input image blocks
output_folder = Path("G:/Project")  # TODO: output masks
output_folder2 = Path("G:/Project")  # TODO: output cropped images
output_folder3 = Path("G:/Project")  # TODO: output marked cropped images
output_folder4 = Path("G:/Project")  # TODO: classified csv files

# ----------------------------------------------------------------------------------------------------------------------
# parameters
# ----------------------------------------------------------------------------------------------------------------------
centroid_pixels_num = 2  # radius of the centroid, used for calculating the centroid color

# ----------------------------------------------------------------------------------------------------------------------
# thresholds
# ----------------------------------------------------------------------------------------------------------------------
centroid_color_threshold = 0.5
std_dev_length_threshold = 5.5
# FakeFC_threshold = 0.2
surrounding_color_threshold1 = 0.025
surrounding_color_threshold2 = 0.045
bbox_ratio_threshold = 0.2
# mean_heteroLength_a1_threshold = -5000
# mean_heteroLength_b1_threshold = -300

# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------
def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm
def calculate_color(b, g, r):
    if float(r) + float(g) == 0:
        return -1
    else:
        return (float(r) - float(g)) / (float(r) + float(g))  # R-G/R+G

    # return 2 * float(g) - float(r) - float(b)
def calculate_color2(color_values):
    b, g, r = np.split(color_values, 3, axis=-1)
    r = r.astype(float)
    g = g.astype(float)
    mask = (r + g) != 0
    index_values = np.full(r.shape, -1.0)
    index_values[mask] = (r[mask] - g[mask]) / (r[mask] + g[mask])

    mean_index_value = np.mean(index_values)
    return mean_index_value
def is_within_crop(position, crop):
    return 0 <= position[0] < crop.shape[0] and 0 <= position[1] < crop.shape[1]
def is_within_base_map(position, base_map):
    return 0 <= position[0] < base_map.shape[0] and 0 <= position[1] < base_map.shape[1]
def find_closest_edge_pixel(centroid, direction, crop, step_size=1):
    if direction is None:
        return None
    current_position = np.array(centroid)
    while True:
        # Round and convert current_position to integers
        current_position_int = np.round(current_position).astype(int)
        if not is_within_crop(current_position_int, crop):
            # If not, move the current position back one step
            current_position = current_position - step_size * direction
            current_position_int = np.round(current_position).astype(int)
            break
        elif crop[current_position_int[0], current_position_int[1], 3] == 0: # Check if the current position is within the crop and if the pixel is not masked
            break
        # Move the current position along the direction
        current_position = current_position + step_size * direction
    return current_position_int
def perform_search(start_position, direction, base_map, image, step_size=1):
    current_position = np.array(start_position)
    length = 0
    color_values = []
    stop_at_start = False

    while True:
        # Move the current position along the direction
        current_position = current_position + step_size * direction
        # Round and convert current_position to integers
        current_position_int = np.round(current_position).astype(int)
        # Check if the current position is within the base map
        if not is_within_base_map(current_position_int, base_map):
            # If not, move the current position back one step
            current_position = current_position - step_size * direction
            if length == 0:  # if it reaches the zero pixel at the beginning -- the length is currently 0
                stop_at_start = True
                break
            break
        elif base_map[current_position_int[0], current_position_int[1]] == 0:  # if reach the zero pixel
            if length == 0:  # if it reaches the zero pixel at the beginning -- the length is currently 0
                stop_at_start = True
                length = 1
                b, g, r = image[current_position_int[0], current_position_int[1]]
                color = calculate_color(b, g, r)
                color_values.append(color)
            break
        # Increase the length
        length += step_size
        # Calculate the color information for the current pixel and add it to the list
        b, g, r = image[current_position_int[0], current_position_int[1]]
        color = calculate_color(b, g, r)
        color_values.append(color)

    # Calculate the mean color_values
    mean_color = np.mean(color_values) if color_values else 0.0  # If color_values is empty, set mean_color to 0

    return length, mean_color, stop_at_start

# ----------------------------------------------------------------------------------------------------------------------
# Workflow
start_time = time.time()
# ······················································································································
# Section 3: Algorithm1, calculate the distribution pattern features of the cropped images
# ······················································································································
# Step 1: find the closest crop and corresponding pixels++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# After cropping the images, calculate the Euclidean distance for each cropped image to each other cropped image
for image_path in input_folder.glob("*.png"):
    # Create a sub-folder for each of image (before/after cropping)
    image_name = image_path.stem
    image_output_folder3 = output_folder3 / image_name
    image_output_folder3.mkdir(parents=True, exist_ok=True)
    image_output_folder4 = output_folder4 / image_name
    image_output_folder4.mkdir(parents=True, exist_ok=True)

    with open(output_folder2 / image_path.stem / 'variables.pkl', 'rb') as f:
        original_crop, original_metadata_records, background_mask = pickle.load(f)

    # read image: OpenCV uses the BGR color space by default, not the RGB space
    image = cv2.imread(str(image_path))  # RGB format input being converted to BGR format, discarding the alpha channel
    pil_image = Image.open(str(image_path))  # read the image with Pillow
    height, width, _ = image.shape

    closest_crops = []
    closest_pixels = []
    centroids = []

    is_full_image_crop = []

    for i, cropped_img_i in enumerate(original_crop):
        # Check the proportion of non-zero pixels in the crop
        non_zero_pixels = np.count_nonzero(cropped_img_i)
        total_pixels = cropped_img_i.size

        # If more than a certain percentage (e.g., 90%) of the pixels are non-zero, skip this crop
        if non_zero_pixels / total_pixels > 0.9:
            print(f"Marking crop {i} because it covers more than 90% of the image")
            is_full_image_crop.append(True)
            closest_crops.append((i, -1))  # -1 indicates that there is no closest crop
            closest_pixels.append((None, None))  # None indicates that there are no closest pixels
            centroids.append(None)
            continue
        else:
            is_full_image_crop.append(False)

        min_distance = np.inf
        closest_crop = None
        closest_pixel_i = None
        closest_pixel_j = None

        # Get the positions of non-zero pixels in the cropped image
        non_zero_i = np.array(np.where(cropped_img_i[:, :, 3] != 0)).T
        centroid_i = np.round(np.mean(non_zero_i, axis=0)).astype(int)
        centroids.append(centroid_i)

        for j, cropped_img_j in enumerate(original_crop):
            if i != j:  # Don't calculate distance between the same crop
                # Get the positions of non-zero pixels in the other cropped image
                non_zero_j = np.array(np.where(cropped_img_j[:, :, 3] != 0)).T

                # Calculate the Euclidean distance between all pairs of non-zero pixels
                tree = KDTree(non_zero_j)
                dist, ind = tree.query(non_zero_i, k=1)

                if np.any(dist == 0):
                    continue

                # Get the minimum distance and its position
                current_min_distance = np.min(dist) if dist.size > 0 else np.inf
                min_position = np.unravel_index(dist.argmin(), dist.shape)

                # If the current minimum distance is less than the global minimum distance, update the global minimum distance and the closest crop
                if current_min_distance < min_distance:
                    min_distance = current_min_distance
                    closest_crop = j

                    closest_pixel_i = non_zero_i[min_position[0]]
                    closest_pixel_j = non_zero_j[ind[min_position[0]]]

        # Store the closest crop and the closest pixels
        closest_crops.append((i, closest_crop if closest_crop is not None else -1))
        closest_pixels.append((closest_pixel_i, closest_pixel_j))  #  the specific pixels from each pair of closest crops that are closest to each other

    # Step 2: calculate the vector from the centroid of each crop to the closest pixel in the same crop+++++++++++++++++
    vectors = []  # list of tuples of (vector, orthogonal_direction_1, orthogonal_direction_2)
    for i in range(len(original_crop)):
        if is_full_image_crop[i]:
            vectors.append((None, None, None, None, None, None, None, None))  # None indicates that there are no vectors
            continue

        centroid_i = centroids[i]
        closest_pixel_i = closest_pixels[i][0]

        if closest_pixel_i is None:
            vectors.append((None, None, None, None, None, None, None, None))  # None indicates that there are no vectors
            continue
        # Calculate eight vectors and convert them to unit vectors
        # The order in a clockwise direction is: vector_i, vector_i_45, orthogonal_direction_1 orthogonal_direction_1_45, opposite_vector_i, opposite_vector_i_45, orthogonal_direction_2, orthogonal_direction_2_45
        vector_i = (closest_pixel_i - centroid_i) / np.linalg.norm(closest_pixel_i - centroid_i)
        if np.isnan(vector_i).any():
            vectors.append((None, None, None, None, None, None, None, None))  # None indicates that there are no vectors
            continue
        opposite_vector_i = (centroid_i - closest_pixel_i) / np.linalg.norm(centroid_i - closest_pixel_i) # opposite vector
        orthogonal_direction_1 = (np.array([-vector_i[1], vector_i[0]])) / np.linalg.norm(np.array([-vector_i[1], vector_i[0]]))  # Calculate two orthogonal directions to this vector
        orthogonal_direction_2 = (np.array([vector_i[1], -vector_i[0]])) / np.linalg.norm(np.array([vector_i[1], -vector_i[0]]))

        rotation_matrix = np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4)], [np.sin(np.pi / 4), np.cos(np.pi / 4)]])
        vector_i_45 = (np.dot(rotation_matrix, vector_i)) /  np.linalg.norm((np.dot(rotation_matrix, vector_i)))  # Use rotation matrix to rotate the vectors by 45 degrees
        opposite_vector_i_45 = (np.dot(rotation_matrix, opposite_vector_i)) / np.linalg.norm(np.dot(rotation_matrix, opposite_vector_i))
        orthogonal_direction_1_45 = (np.dot(rotation_matrix, orthogonal_direction_1)) / np.linalg.norm(np.dot(rotation_matrix, orthogonal_direction_1))
        orthogonal_direction_2_45 = (np.dot(rotation_matrix, orthogonal_direction_2)) / np.linalg.norm(np.dot(rotation_matrix, orthogonal_direction_2))

        vectors.append((vector_i, opposite_vector_i, orthogonal_direction_1, orthogonal_direction_2, vector_i_45,
                        opposite_vector_i_45, orthogonal_direction_1_45,
                        orthogonal_direction_2_45))  # Store the vectors and the orthogonal directions

    # Step 3-1: Search the started pixels for each crop, perform the searching procedure, and record the lengths++++++++++
    base_map = background_mask.copy()  # Create the base map
    base_map_crop = cv2.bitwise_and(image, image, mask=background_mask)  # crop the entire image, but in BGRA format

    search_lengths = []  # Initialize the lengths of the searches
    mean_color_centroid = []
    mean_colors = []
    mean_colors_mean_list = []

    connect_lengths = []
    connect_directions = []
    connect_angles = []

    stop_at_start_counts = []

    for i in range(len(original_crop)):
        if is_full_image_crop[i]:
            # Generate placeholder data for full-image crops
            search_lengths.append((None, None, None, None, None, None, None, None))  # None indicates that there are no lengths
            mean_color_centroid.append(None)  # None indicates that there is no mean color
            mean_colors.append((None, None, None, None, None, None, None, None))  # None indicates that there are no mean colors
            mean_colors_mean_list.append(None)  # None indicates that there is no mean color
            connect_lengths.append([None, None, None, None, None, None, None, None])  # None indicates that there are no lengths
            connect_directions.append([None, None, None, None, None, None, None, None])  # None indicates that there are no directions
            connect_angles.append([None, None, None, None, None, None, None, None])  # None indicates that there are no angles
            stop_at_start_counts.append(None)  # None indicates that there is no count
            continue
        # Get the color information of the centroid ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        centroid_i = centroids[i]
        start_row = max(0, centroid_i[0] - centroid_pixels_num)
        end_row = min(original_crop[i].shape[0], centroid_i[0] + centroid_pixels_num + 1)
        start_col = max(0, centroid_i[1] - centroid_pixels_num)
        end_col = min(original_crop[i].shape[1], centroid_i[1] + centroid_pixels_num + 1)

        color_values = original_crop[i][start_row:end_row, start_col:end_col, :3]  # Get the color values of the pixels around the centroid

        mask_values = original_crop[i][start_row:end_row, start_col:end_col, 3]
        color_values = color_values[mask_values != 0, :]

        # Calculate the color information for each pixel and take the mean
        mean_color_centroid_i = calculate_color2(color_values)
        mean_color_centroid.append(mean_color_centroid_i)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # convert the vectors, get the started pixel, and perform the search to get the lengths ~~~~~~~~~~~~~~~~~~~~~~~~
        vector_i, opposite_vector_i, orthogonal_direction_1, orthogonal_direction_2, vector_i_45, opposite_vector_i_45, orthogonal_direction_1_45, orthogonal_direction_2_45 = vectors[i]

        # Find the eight pixels that are closest to the edge of the crop along the three directions
        edge_pixel_i = closest_pixels[i][0] if closest_pixels[i][0] is not None else None
        edge_pixel_opposite_i = find_closest_edge_pixel(centroid_i, opposite_vector_i, original_crop[i]) if opposite_vector_i is not None else None
        edge_pixel_orthogonal_1 = find_closest_edge_pixel(centroid_i, orthogonal_direction_1, original_crop[i]) if orthogonal_direction_1 is not None else None
        edge_pixel_orthogonal_2 = find_closest_edge_pixel(centroid_i, orthogonal_direction_2, original_crop[i]) if orthogonal_direction_2 is not None else None
        edge_pixel_i_45 = find_closest_edge_pixel(centroid_i, vector_i_45,original_crop[i]) if vector_i_45 is not None else None
        edge_pixel_opposite_i_45 = find_closest_edge_pixel(centroid_i, opposite_vector_i_45, original_crop[i]) if opposite_vector_i_45 is not None else None
        edge_pixel_orthogonal_1_45 = find_closest_edge_pixel(centroid_i, orthogonal_direction_1_45, original_crop[i]) if orthogonal_direction_1_45 is not None else None
        edge_pixel_orthogonal_2_45 = find_closest_edge_pixel(centroid_i, orthogonal_direction_2_45, original_crop[i]) if orthogonal_direction_2_45 is not None else None

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Mark these pixels on the original_crop image

        marked_crop = original_crop[i].copy()
        marker_size = 3  # Set the size of the marker
        if edge_pixel_i is not None:
            # Red color for edge_pixel_i
            marked_crop[edge_pixel_i[0] - marker_size:edge_pixel_i[0] + marker_size,
            edge_pixel_i[1] - marker_size:edge_pixel_i[1] + marker_size, :3] = [255, 0, 0]

        if edge_pixel_opposite_i is not None:
            # Green color for edge_pixel_opposite_i
            marked_crop[edge_pixel_opposite_i[0] - marker_size:edge_pixel_opposite_i[0] + marker_size,
            edge_pixel_opposite_i[1] - marker_size:edge_pixel_opposite_i[1] + marker_size, :3] = [0, 255, 0]

        if edge_pixel_orthogonal_1 is not None:
            # Blue color for edge_pixel_orthogonal_1
            marked_crop[edge_pixel_orthogonal_1[0] - marker_size:edge_pixel_orthogonal_1[0] + marker_size,
            edge_pixel_orthogonal_1[1] - marker_size:edge_pixel_orthogonal_1[1] + marker_size, :3] = [0, 0, 255]

        if edge_pixel_orthogonal_2 is not None:
            # Yellow color for edge_pixel_orthogonal_2
            marked_crop[edge_pixel_orthogonal_2[0] - marker_size:edge_pixel_orthogonal_2[0] + marker_size,
            edge_pixel_orthogonal_2[1] - marker_size:edge_pixel_orthogonal_2[1] + marker_size, :3] = [255, 255, 0]

        if centroid_i is not None:
            marked_crop[centroid_i[0] - marker_size:centroid_i[0] + marker_size,
            centroid_i[1] - marker_size:centroid_i[1] + marker_size, :3] = [255, 255, 255]

        cv2.imwrite(str(image_output_folder3 / f"marked_crop_{i + 1}_{image_name}.png"), marked_crop) # [B G R Cyan]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Perform the searches and record the lengths
        search_length_i, mean_color_i, stop_at_start_i = perform_search(edge_pixel_i, vector_i, base_map,image) if vector_i is not None else (None, None, None)
        search_length_opposite_i, mean_color_opposite_i, stop_at_start_opposite_i = perform_search(edge_pixel_opposite_i, opposite_vector_i, base_map, image) if opposite_vector_i  is not None else (None, None, None)
        search_length_orthogonal_1, mean_color_orthogonal_1, stop_at_start_orthogonal_1 = perform_search(edge_pixel_orthogonal_1, orthogonal_direction_1, base_map, image) if orthogonal_direction_1 is not None else (None, None, None)
        search_length_orthogonal_2, mean_color_orthogonal_2, stop_at_start_orthogonal_2 = perform_search(edge_pixel_orthogonal_2, orthogonal_direction_2, base_map, image) if orthogonal_direction_2 is not None else (None, None, None)
        search_length_i_45, mean_color_i_45, stop_at_start_length_i_45 = perform_search(edge_pixel_i_45, vector_i_45, base_map, image) if vector_i_45 is not None else (None, None, None)
        search_length_opposite_i_45, mean_color_opposite_i_45, stop_at_start_opposite_i_45 = perform_search(edge_pixel_opposite_i_45, opposite_vector_i_45, base_map, image) if opposite_vector_i_45 is not None else (None, None, None)
        search_length_orthogonal_1_45, mean_color_orthogonal_1_45, stop_at_start_orthogonal_1_45 = perform_search(edge_pixel_orthogonal_1_45, orthogonal_direction_1_45, base_map, image) if orthogonal_direction_1_45 is not None else (None, None, None)
        search_length_orthogonal_2_45, mean_color_orthogonal_2_45, stop_at_start_orthogonal_2_45 = perform_search(edge_pixel_orthogonal_2_45, orthogonal_direction_2_45, base_map, image) if orthogonal_direction_2_45 is not None else (None, None, None)

        stop_at_start_values = [stop_at_start_i, stop_at_start_opposite_i, stop_at_start_orthogonal_1, stop_at_start_orthogonal_2, stop_at_start_length_i_45, stop_at_start_opposite_i_45,
                                stop_at_start_orthogonal_1_45, stop_at_start_orthogonal_2_45]
        stop_at_start_values = [value for value in stop_at_start_values if value is not None]
        stop_at_start_count = sum(stop_at_start_values)
        stop_at_start_counts.append(stop_at_start_count)

        # Store the search lengths
        search_lengths_i = (search_length_i, search_length_opposite_i, search_length_orthogonal_1, search_length_orthogonal_2,
                            search_length_i_45, search_length_opposite_i_45, search_length_orthogonal_1_45, search_length_orthogonal_2_45)
        mean_colors_i = (mean_color_i, mean_color_opposite_i, mean_color_orthogonal_1, mean_color_orthogonal_2,
                         mean_color_i_45, mean_color_opposite_i_45, mean_color_orthogonal_1_45, mean_color_orthogonal_2_45)
        mean_colors_i = [color for color in mean_colors_i if color is not None]
        mean_colors_i_mean = np.mean(mean_colors_i, axis=0) if mean_colors_i else None

        search_lengths.append(search_lengths_i)
        mean_colors.append(mean_colors_i)
        mean_colors_mean_list.append(mean_colors_i_mean)

        # connect all the edge pixels we found previously in turn, and record the length, direction of the connected line
        # Retrieve the edge pixels in the order of vectors

        edge_pixels = [
            closest_pixels[i][0] if closest_pixels[i][0] is not None else None,  # edge_pixel_i
            find_closest_edge_pixel(centroid_i, vector_i_45, original_crop[i]) if vector_i_45 is not None else None, # edge_pixel_i_45
            find_closest_edge_pixel(centroid_i, orthogonal_direction_1, original_crop[i]) if orthogonal_direction_1 is not None else None,  # edge_pixel_orthogonal_1
            find_closest_edge_pixel(centroid_i, orthogonal_direction_1_45, original_crop[i]) if orthogonal_direction_1_45 is not None else None,  # edge_pixel_orthogonal_1_45
            find_closest_edge_pixel(centroid_i, opposite_vector_i, original_crop[i]) if opposite_vector_i is not None else None,  # edge_pixel_opposite_i
            find_closest_edge_pixel(centroid_i, opposite_vector_i_45, original_crop[i]) if opposite_vector_i_45 is not None else None,  # edge_pixel_opposite_i_45
            find_closest_edge_pixel(centroid_i, orthogonal_direction_2, original_crop[i]) if orthogonal_direction_2 is not None else None,  # edge_pixel_orthogonal_2
            find_closest_edge_pixel(centroid_i, orthogonal_direction_2_45, original_crop[i]) if orthogonal_direction_2_45 is not None else None  # edge_pixel_orthogonal_2_45
        ]
        connect_length_group = []
        connect_direction_group = []
        connect_angle_group = []

        # Calculate the lengths, directions, and angles
        for j in range(len(edge_pixels)):
            if edge_pixels[j] is None or edge_pixels[(j + 1) % len(edge_pixels)] is None:
                connect_length_group.append(None)
                connect_direction_group.append(None)
                connect_angle_group.append(None)
                # Skip this iteration
                continue
            # Calculate the length of the connect line
            next_j = (j + 1) % len(edge_pixels)  # This will be 0 when j is the last index
            connect_length = np.linalg.norm(edge_pixels[next_j] - edge_pixels[j])
            connect_length_group.append(connect_length)

            # Calculate the direction of the connect line
            if connect_length != 0:  # Avoid division by zero
                connect_direction = (edge_pixels[next_j] - edge_pixels[j]) / connect_length  # unit vector
            else:
                connect_direction = np.array([0, 0])  # Or whatever is appropriate in your context
            connect_direction_group.append(connect_direction)

            # Calculate the angle between the connect line and the two corresponding vectors
            # We use the dot product to calculate the angle: cos(theta) = dot(a, b) / (||a|| ||b||)
            # And then use arccos to get the angle in radians, and convert it to degrees
            if j == 0:  # For the first connect line
                normalized_connect_direction = normalize(-connect_direction)
                normalized_vector_i = normalize(vector_i)
                dot_product = np.clip(np.dot(normalized_connect_direction, normalized_vector_i), -1, 1)
                angle_with_vector_i = np.arccos(dot_product) * 180 / np.pi

                normalized_vector_i_45 = normalize(vector_i_45)
                dot_product = np.clip(np.dot(normalized_connect_direction, normalized_vector_i_45), -1, 1)
                angle_with_vector_i_45 = np.arccos(dot_product) * 180 / np.pi
            else:  # For the other connect lines
                normalized_connect_direction = normalize(connect_direction)
                normalized_vector_i = normalize(vectors[i][j])
                dot_product = np.clip(np.dot(normalized_connect_direction, normalized_vector_i), -1, 1)
                angle_with_vector_i = np.arccos(dot_product) * 180 / np.pi

                normalized_vector_i_45 = normalize(vectors[i][next_j])
                dot_product = np.clip(np.dot(normalized_connect_direction, normalized_vector_i_45), -1, 1)
                angle_with_vector_i_45 = np.arccos(dot_product) * 180 / np.pi

            connect_angle_group.append((angle_with_vector_i, angle_with_vector_i_45))

        connect_lengths.append(connect_length_group)
        connect_directions.append(connect_direction_group)
        connect_angles.append(connect_angle_group)

    # sum_of_lengths = [sum(lengths) for lengths in search_lengths]  # Calculate the sum of lengths for each tuple


    # Step 3-2: Calculate the heterogeneity Length for a1_crop +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    sum_heterogeneity_length = []
    mean_heterogeneity_length = []
    heterogeneity_list = []
    for id, crop, lengths, stop_at_start_num, centroid_color, pixel_colors, mean_color_mean, connect_length, connect_direction, connect_angle,record \
            in zip(range(len(original_crop)), original_crop, search_lengths, stop_at_start_counts, mean_color_centroid, mean_colors, mean_colors_mean_list,
                   connect_lengths, connect_directions, connect_angles, original_metadata_records):
        if is_full_image_crop[id]:
            # Generate placeholder data for full-image crops
            sum_heterogeneity_length.append(None)  # None indicates that there is no sum of heterogeneity lengths
            mean_heterogeneity_length.append(None)  # None indicates that there is no mean of heterogeneity lengths
            heterogeneity_list.append([id] + [None] * 24)  # None indicates that there is no data
            continue

        if len(pixel_colors) != 8:
            continue
        search_length_i, search_length_opposite_i, search_length_orthogonal_1, search_length_orthogonal_2, \
            search_length_i_45, search_length_opposite_i_45, search_length_orthogonal_1_45, search_length_orthogonal_2_45 = lengths
        mean_color_i, mean_color_opposite_i, mean_color_orthogonal_1, mean_color_orthogonal_2, \
            mean_color_i_45, mean_color_opposite_i_45, mean_color_orthogonal_1_45, mean_color_orthogonal_2_45 = pixel_colors

        # Calculate the heterogeneity length
        heterogeneity_length_i = search_length_i * (mean_color_i - centroid_color)
        heterogeneity_length_opposite_i = search_length_opposite_i * (mean_color_opposite_i - centroid_color)
        heterogeneity_length_orthogonal_1 = search_length_orthogonal_1 * (mean_color_orthogonal_1 - centroid_color)
        heterogeneity_length_orthogonal_2 = search_length_orthogonal_2 * (mean_color_orthogonal_2 - centroid_color)
        heterogeneity_length_i_45 = search_length_i_45 * (mean_color_i_45 - centroid_color)
        heterogeneity_length_opposite_i_45 = search_length_opposite_i_45 * (mean_color_opposite_i_45 - centroid_color)
        heterogeneity_length_orthogonal_1_45 = search_length_orthogonal_1_45 * (mean_color_orthogonal_1_45 - centroid_color)
        heterogeneity_length_orthogonal_2_45 = search_length_orthogonal_2_45 * (mean_color_orthogonal_2_45 - centroid_color)

        sum_heterogeneity_length_i = sum([heterogeneity_length_i, heterogeneity_length_opposite_i, heterogeneity_length_orthogonal_1, heterogeneity_length_orthogonal_2,
                                          heterogeneity_length_i_45, heterogeneity_length_opposite_i_45, heterogeneity_length_orthogonal_1_45, heterogeneity_length_orthogonal_2_45])
        mean_heterogeneity_length_i = sum_heterogeneity_length_i / 8
        sum_heterogeneity_length.append(sum_heterogeneity_length_i)
        mean_heterogeneity_length.append(mean_heterogeneity_length_i)

        heterogeneity_list.append([id, search_length_i, search_length_opposite_i, search_length_orthogonal_1, search_length_orthogonal_2,
                                   search_length_i_45, search_length_opposite_i_45, search_length_orthogonal_1_45, search_length_orthogonal_2_45, stop_at_start_num,
                                   mean_color_i, mean_color_opposite_i, mean_color_orthogonal_1, mean_color_orthogonal_2,
                                   mean_color_i_45, mean_color_opposite_i_45, mean_color_orthogonal_1_45, mean_color_orthogonal_2_45, mean_color_mean,
                                   centroid_color, sum_heterogeneity_length_i, mean_heterogeneity_length_i, connect_length, connect_direction, connect_angle])

    # # Separate the heterogeneity list for a1_crop and b1_crop
    # heterogeneity_list_a1 = [heterogeneity_list[i] for i in a1_ids]
    # heterogeneity_list_b1 = [heterogeneity_list[i] for i in b1_ids]

    # Write data to .csv files
    with open(image_output_folder4 / f"heterogeneity_{image_name}.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "search_length_i", "search_length_opposite_i", "search_length_orthogonal_1", "search_length_orthogonal_2",
                         'search_length_i_45', 'search_length_opposite_i_45', 'search_length_orthogonal_1_45', 'search_length_orthogonal_2_45', 'stop_at_start_counts',
                         "mean_color_i", "mean_color_opposite_i", "mean_color_orthogonal_1", "mean_color_orthogonal_2",
                         "mean_color_i_45", "mean_color_opposite_i_45", "mean_color_orthogonal_1_45", "mean_color_orthogonal_2_45", "mean_color_mean", "centroid_color",
                         "sum_heterogeneity_length", "mean_heterogeneity_length", "connect_length", "connect_direction", "connect_angle"])
        writer.writerows(heterogeneity_list)

    # ··················································································································
    # Classification
    # ··················································································································
    crop_classified1 = []
    a1_crop = []
    a2_crop = []
    b1_crop = []

    # mean_heterogeneity_length_i = [row[11] for row in heterogeneity_list]
    # threshold_a2 = otsu_threshold(mean_heterogeneity_length_i)  # Calculate the Otsu's threshold
    # # kmeans = KMeans(n_clusters=3, random_state=0).fit(np.array(mean_heterogeneity_length_i).reshape(-1, 1))
    # # threshold_a1 = np.mean(kmeans.cluster_centers_)

    # Classification 1: Gradient direction++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for row, crop in zip(heterogeneity_list, original_crop):
        id = row[0]
        if is_full_image_crop[id]:
            # Generate placeholder data for full-image crops
            crop_classified1.append(row + [None])  # None indicates that there is no classification
            continue
        id, search_length_i, search_length_opposite_i, search_length_orthogonal_1, search_length_orthogonal_2, \
            search_length_i_45, search_length_opposite_i_45, search_length_orthogonal_1_45, search_length_orthogonal_2_45, stop_at_start_counts, \
            mean_color_i, mean_color_opposite_i, mean_color_orthogonal_1, mean_color_orthogonal_2, \
            mean_color_i_45, mean_color_opposite_i_45, mean_color_orthogonal_1_45, mean_color_orthogonal_2_45, mean_color_mean, centroid_color, \
            sum_heterogeneity_length_i, mean_heterogeneity_length_i, connect_length, connect_direction, connect_angle = row

        if mean_color_mean > centroid_color:
            if centroid_color < centroid_color_threshold:
                crop_classified1.append(row + ['a2'])  # Vegetation
                a2_crop.append(crop)
            else:
                crop_classified1.append(row + ['a1'])  # Tidal flat
                a1_crop.append(crop)
        elif mean_color_mean <= centroid_color:
            crop_classified1.append(row + ['b1'])  # Tidal flat or Withered Vegetation
            b1_crop.append(crop)

    # Merge and save the crops of a2 and a3, b2 and b3 to the folder
    if len(a1_crop) > 0:
        a1_merged = np.zeros_like(a1_crop[0])
        for crop in a1_crop:
            a1_merged = np.maximum(a1_merged, crop)
        cv2.imwrite(str(image_output_folder4 / f"a1_merged_{image_name}.png"), a1_merged)

    if len(a2_crop) > 0:
        a2_merged = np.zeros_like(a2_crop[0])
        for crop in a2_crop:
            a2_merged = np.maximum(a2_merged, crop)
        cv2.imwrite(str(image_output_folder4 / f"a2_merged_{image_name}.png"), a2_merged)

    if len(b1_crop) > 0:
        b1_merged = np.zeros_like(b1_crop[0])
        for crop in b1_crop:
            b1_merged = np.maximum(b1_merged, crop)
        cv2.imwrite(str(image_output_folder4 / f"b1_merged_{image_name}.png"), b1_merged)

    # # Write the classified data to .csv files
    # with open(image_output_folder4 / f"crop_classified1_{image_name}.csv", 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["id", "search_length_i", "search_length_opposite_i", "search_length_orthogonal_1", "search_length_orthogonal_2",
    #                      'search_length_i_45', 'search_length_opposite_i_45', 'search_length_orthogonal_1_45', 'search_length_orthogonal_2_45', 'stop_at_start_counts',
    #                      "mean_color_i", "mean_color_opposite_i", "mean_color_orthogonal_1", "mean_color_orthogonal_2",
    #                      "mean_color_i_45", "mean_color_opposite_i_45", "mean_color_orthogonal_1_45", "mean_color_orthogonal_2_45", "mean_color_mean", "centroid_color",
    #                      "sum_heterogeneity_length", "mean_heterogeneity_length", "connect_length", "connect_direction", "connect_angle", "class1"])
    #     writer.writerows(crop_classified1)

    # Classification2: Stress analysis++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # standard deviation of the connect length, connect direction, and connect angle
    df_bbox = pd.read_csv(output_folder2 / image_path.stem / f"metadata_{image_name}.csv")
    bbox_w_dict = df_bbox.set_index('id')['bbox_w'].to_dict()
    bbox_h_dict = df_bbox.set_index('id')['bbox_h'].to_dict()

    std_dev_lengths = []
    std_dev_directions = []
    std_dev_angles = []

    # Calculate standard deviations
    for i in range(len(original_crop)):
        if any(elem is None for elem in connect_lengths[i]):
            std_dev_lengths.append(None)
        else:
            std_dev_lengths.append(np.std(connect_lengths[i]))

        if any(elem is None for elem in connect_directions[i]):
            std_dev_directions.append(None)
        else:
            std_dev_directions.append(np.std(connect_directions[i]))

        if any(elem is None for elem in connect_angles[i]):
            std_dev_angles.append(None)
        else:
            std_dev_angles.append(np.std(connect_angles[i]))

    crop_classified2 = []
    for i, row in enumerate(crop_classified1):
        id = row[0]
        if is_full_image_crop[id]:
            # Generate placeholder data for full-image crops
            crop_classified2.append(row + [None, None, None, None])  # None indicates that there is no classification
            continue
        std_dev_length = std_dev_lengths[i]
        std_dev_direction = std_dev_directions[i]
        std_dev_angle = std_dev_angles[i]
        class1 = row[-1]  # Get the class from crop_classified1

        bbox_w_ratio = bbox_w_dict[id] / width
        bbox_h_ratio = bbox_h_dict[id] / height

        # Apply the filter rule for a2 and b1
        if class1 == 'a2':
            if std_dev_length is None:
                class2 = None
            elif std_dev_length < std_dev_length_threshold and (bbox_w_ratio < bbox_ratio_threshold or bbox_h_ratio < bbox_ratio_threshold):
                class2 = 'a4'
            else:
                class2 = 'a3'
        elif class1 == 'b1':
            if std_dev_length is None:
                class2 = None
            elif std_dev_length < std_dev_length_threshold and (bbox_w_ratio < bbox_ratio_threshold or bbox_h_ratio < bbox_ratio_threshold):
                class2 = 'b3'
            else:
                class2 = 'b2'
        else:
            class2 = class1  # Keep the class from crop_classified2 for the other crops

        new_row = row + [std_dev_length, std_dev_direction, std_dev_angle, class2]  # Append class2 to the row
        crop_classified2.append(new_row)

    # # Write the new classified data to a CSV file
    # with open(image_output_folder4 / f"crop_classified2_{image_name}.csv", 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["id", "search_length_i", "search_length_opposite_i", "search_length_orthogonal_1",
    #                      "search_length_orthogonal_2",
    #                      'search_length_i_45', 'search_length_opposite_i_45', 'search_length_orthogonal_1_45',
    #                      'search_length_orthogonal_2_45', 'stop_at_start_counts',
    #                      "mean_color_i", "mean_color_opposite_i", "mean_color_orthogonal_1", "mean_color_orthogonal_2",
    #                      "mean_color_i_45", "mean_color_opposite_i_45", "mean_color_orthogonal_1_45",
    #                      "mean_color_orthogonal_2_45", "mean_color_mean", "centroid_color",
    #                      "sum_heterogeneity_length", "mean_heterogeneity_length", "connect_length", "connect_direction",
    #                      "connect_angle",  "class1", "std_dev_length", "std_dev_direction", "std_dev_angle", "class2"])
    #     writer.writerows(crop_classified2)

    a3_crop = []
    a4_crop = []
    b2_crop = []
    b3_crop = []

    # Append crops to their corresponding lists based on their class in crop_classified2
    for row, crop in zip(crop_classified2, original_crop):
        class2 = row[-1]  # Get the class from crop_classified2
        if class2 == 'a3':
            a3_crop.append(crop)
        elif class2 == 'a4':
            a4_crop.append(crop)
        elif class2 == 'b2':
            b2_crop.append(crop)
        elif class2 == 'b3':
            b3_crop.append(crop)

    # Merge and save the crops of each new class to the folder
    if len(a3_crop) > 0:
        a3_merged = np.zeros_like(a3_crop[0])
        for crop in a3_crop:
            a3_merged = np.maximum(a3_merged, crop)
        cv2.imwrite(str(image_output_folder4 / f"a3_merged_{image_name}.png"), a3_merged)

    if len(a4_crop) > 0:
        a4_merged = np.zeros_like(a4_crop[0])
        for crop in a4_crop:
            a4_merged = np.maximum(a4_merged, crop)
        cv2.imwrite(str(image_output_folder4 / f"a4_merged_{image_name}.png"), a4_merged)

    if len(b2_crop) > 0:
        b2_merged = np.zeros_like(b2_crop[0])
        for crop in b2_crop:
            b2_merged = np.maximum(b2_merged, crop)
        cv2.imwrite(str(image_output_folder4 / f"b2_merged_{image_name}.png"), b2_merged)

    if len(b3_crop) > 0:
        b3_merged = np.zeros_like(b3_crop[0])
        for crop in b3_crop:
            b3_merged = np.maximum(b3_merged, crop)
        cv2.imwrite(str(image_output_folder4 / f"b3_merged_{image_name}.png"), b3_merged)


    # Classification 3: Mobility and Color of Surrounding+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    crop_classified3 = []
    # Append crops to their corresponding lists based on their class in crop_classified2 and the new filtering rule
    for row in crop_classified2:
        id = row[0]
        if is_full_image_crop[id]:
            # Generate placeholder data for full-image crops
            crop_classified3.append(row + [None])
            continue
        class2 = row[-1]  # Get the class from crop_classified2
        # lengths = row[1:9]
        stop_at_start_counts = row[9]  # Adjust the indices if necessary
        mean_color_mean = row[18]
        centroid_color = row[19]
        mean_colors = row[10:18]
        mean_colors_greater_than_zero = sum(1 for color in mean_colors if color is not None and color > 0)
        # Apply the new filtering rule

        if class2 == 'a4':
            if stop_at_start_counts >= 5 and mean_color_mean < surrounding_color_threshold1: #and mean_colors_greater_than_zero <= 3:
                class3 = 'a5'
            else:
                class3 = 'a6'
        elif class2 == 'b3':
            if stop_at_start_counts >= 5 and mean_color_mean < surrounding_color_threshold2: # and mean_colors_greater_than_zero <= 3:
                class3 = 'b4'
            else:
                class3 = 'b5'
        else:
            class3 = class2  # Keep the class from crop_classified2 for the other crops

        new_row = row + [class3]
        crop_classified3.append(new_row)

    # # Write the new classified data to a CSV file
    # with open(image_output_folder4 / f"crop_classified3_{image_name}.csv", 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["id", "search_length_i", "search_length_opposite_i", "search_length_orthogonal_1",
    #                      "search_length_orthogonal_2",
    #                      'search_length_i_45', 'search_length_opposite_i_45', 'search_length_orthogonal_1_45',
    #                      'search_length_orthogonal_2_45', 'stop_at_start_counts',
    #                      "mean_color_i", "mean_color_opposite_i", "mean_color_orthogonal_1",
    #                      "mean_color_orthogonal_2",
    #                      "mean_color_i_45", "mean_color_opposite_i_45", "mean_color_orthogonal_1_45",
    #                      "mean_color_orthogonal_2_45", "mean_color_mean", "centroid_color",
    #                      "sum_heterogeneity_length", "mean_heterogeneity_length", "connect_length",
    #                      "connect_direction", "connect_angle", "class1", "std_dev_length", "std_dev_direction", "std_dev_angle", "class2", "class3"])
    #     writer.writerows(crop_classified3)

    a5_crop = []
    a6_crop = []
    b4_crop = []
    b5_crop = []

    # Separate the crops based on their class in crop_classified3
    for row, crop in zip(crop_classified3, original_crop):
        class3 = row[-1]  # Get the class from crop_classified3
        if class3 == 'a5':
            a5_crop.append(crop)
        elif class3 == 'a6':
            a6_crop.append(crop)
        elif class3 == 'b4':
            b4_crop.append(crop)
        elif class3 == 'b5':
            b5_crop.append(crop)

    # Merge and save the crops of a5 and a6 to the folder+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if len(a5_crop) > 0:
        a5_merged = np.zeros_like(a5_crop[0])
        for crop in a5_crop:
            a5_merged = np.maximum(a5_merged, crop)
        cv2.imwrite(str(image_output_folder4 / f"a5_merged_{image_name}.png"), a5_merged)

    if len(a6_crop) > 0:
        a6_merged = np.zeros_like(a6_crop[0])
        for crop in a6_crop:
            a6_merged = np.maximum(a6_merged, crop)
        cv2.imwrite(str(image_output_folder4 / f"a6_merged_{image_name}.png"), a6_merged)

    if len(b4_crop) > 0:
        b4_merged = np.zeros_like(b4_crop[0])
        for crop in b4_crop:
            b4_merged = np.maximum(b4_merged, crop)
        cv2.imwrite(str(image_output_folder4 / f"b4_merged_{image_name}.png"), b4_merged)

    if len(b5_crop) > 0:
        b5_merged = np.zeros_like(b5_crop[0])
        for crop in b5_crop:
            b5_merged = np.maximum(b5_merged, crop)
        cv2.imwrite(str(image_output_folder4 / f"b5_merged_{image_name}.png"), b5_merged)

    # Save the Classification 3 metadata, and merge it to the previous metadata
    df = pd.read_csv(output_folder2 / image_path.stem / f"metadata_{image_name}.csv")  # the previous metadata includes Label

    df_classified3 = pd.DataFrame(crop_classified3,
                                  columns=["id", "search_length_i", "search_length_opposite_i",
                                           "search_length_orthogonal_1",
                                           "search_length_orthogonal_2",
                                           'search_length_i_45', 'search_length_opposite_i_45',
                                           'search_length_orthogonal_1_45',
                                           'search_length_orthogonal_2_45', 'stop_at_start_counts',
                                           "mean_color_i", "mean_color_opposite_i", "mean_color_orthogonal_1",
                                           "mean_color_orthogonal_2",
                                           "mean_color_i_45", "mean_color_opposite_i_45",
                                           "mean_color_orthogonal_1_45",
                                           "mean_color_orthogonal_2_45", "mean_color_mean", "centroid_color",
                                           "sum_heterogeneity_length", "mean_heterogeneity_length",
                                           "connect_length",
                                           "connect_direction", "connect_angle", "class1", "std_dev_length",
                                           "std_dev_direction", "std_dev_angle", "class2", "class3"])

    df_classified3.to_csv(image_output_folder4 / f"crop_classified3_{image_name}.csv", index=False)
    df_final = pd.merge(df, df_classified3, on="id") # Use id to Merge the previous metadata with the new classified data
    df_final = df_final[
        [c for c in df_final if c not in ['class1', 'class2', 'class3']] + ['class1', 'class2', 'class3']]   # Move the class columns to the end
    df_final.to_csv(image_output_folder4 / f"final_classified_metadata_{image_name}.csv", index=False)

# ----------------------------------------------------------------------------------------------------------------------

end_time = time.time()
elapsed_time_minutes = (time.time() - start_time) / 60
print(f"Elapsed time: {elapsed_time_minutes} seconds")