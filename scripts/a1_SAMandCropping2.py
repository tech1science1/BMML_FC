# segmenting the images, generate the masks and cropped images, calculate the additional features

import cv2
import numpy as np
import torch
import time
import colorsys
import csv
import pickle
from PIL import Image, ExifTags
from pathlib import Patha1_SAMandCropping2.py
from scipy import ndimage
from scipy.spatial import distance

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


# Model and Path
model_path = Path('G:/Project')  # TODO:SAM checkpoint
input_folder = Path('G:/Project')  # TODO: input images
output_folder = Path("G:/Project/")  # TODO: output masks
output_folder2 = Path("G:/Project/")  # TODO: output cropped images


# ----------------------------------------------------------------------------------------------------------------------
# load checkpoint & use cuda
# ----------------------------------------------------------------------------------------------------------------------
device = torch.device("cuda")
sam = sam_model_registry["vit_h"](checkpoint=model_path)
sam.to(device=device)
# Ensure output directories exist
output_folder.mkdir(parents=True, exist_ok=True)
output_folder2.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------
def generate_colors(num_colors):
    HSV_tuples = [(x * 1.0 / num_colors, 0.5, 0.5) for x in range(num_colors)]
    RGB_tuples = [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]
    # Convert RGB to BGR and scale to [0,255]
    BGR_tuples = [(int(b * 255), int(g * 255), int(r * 255)) for r, g, b in RGB_tuples]
    return BGR_tuples
# some RGB-based indices recommended by the article "RGB Indices and Canopy Height Modelling for Mapping Tidal Marsh Biomass from a Small Unmanned Aerial System"
def calculate_indices(mean_color_b, mean_color_g, mean_color_r):
    indices = {}

    indices['ExG'] = 2 * mean_color_g - mean_color_r - mean_color_b
    indices['ExGR'] = 2 * mean_color_g - mean_color_r - mean_color_b - (1.4 * mean_color_r - mean_color_g)
    indices['GLI'] = (2 * mean_color_g - mean_color_r - mean_color_b) / (2 * mean_color_g + mean_color_r + mean_color_b) if 2 * mean_color_g + mean_color_r + mean_color_b != 0 else -1
    indices['GCC'] = mean_color_g / (mean_color_b + mean_color_g + mean_color_r) if mean_color_b + mean_color_g + mean_color_r != 0 else -1
    indices['GRVI'] = (mean_color_g - mean_color_r) / (mean_color_g + mean_color_r) if mean_color_g + mean_color_r != 0 else -1
    indices['IKAW'] = (mean_color_r - mean_color_b) / (mean_color_r + mean_color_b) if mean_color_r + mean_color_b != 0 else -1
    indices['MGRVI'] = (mean_color_g ** 2 - mean_color_r ** 2) / (mean_color_g  ** 2 + mean_color_r ** 2) if mean_color_g ** 2 + mean_color_r ** 2 != 0 else -1
    indices['MVARI'] = (mean_color_g - mean_color_b) / (mean_color_g + mean_color_r - mean_color_b) if mean_color_g + mean_color_r - mean_color_b != 0 else -1
    indices['RGBVI'] = (mean_color_g ** 2 - mean_color_b * mean_color_r) / (mean_color_g ** 2 + mean_color_b * mean_color_r) if mean_color_g ** 2 + mean_color_b * mean_color_r != 0 else -1
    indices['TGI'] = mean_color_g - (0.39 * mean_color_r) - (0.61 * mean_color_b)
    indices['VARI'] = (mean_color_g - mean_color_r) / (mean_color_g + mean_color_r - mean_color_b) if mean_color_g + mean_color_r - mean_color_b != 0 else -1
    indices['VDVI'] = (2 * mean_color_g - mean_color_r - mean_color_b) / (2 * mean_color_g + mean_color_r + mean_color_b) if 2 * mean_color_g + mean_color_r + mean_color_b != 0 else -1
    indices['NGRDI'] = (mean_color_g - mean_color_r) / (mean_color_g + mean_color_r) if mean_color_g + mean_color_r != 0 else -1


    return indices

def calculate_indices_pixel(image):
    b, g, r = cv2.split(image)
    # Convert the channels to float32 to avoid overflow and underflow
    b = b.astype(np.float32)
    g = g.astype(np.float32)
    r = r.astype(np.float32)

    # denominator = g ** 2 + b * r
    # mask_notzero = denominator != 0  # Create a mask where the denominator is not zero
    # rgbvi = np.full_like(b, -1.0, dtype=np.float32)  # Initialize an array for the 'RGBVI' index with -1.0
    #
    # rgbvi[mask_notzero] = (g[mask_notzero] ** 2 - b[mask_notzero] * r[mask_notzero]) / denominator[mask_notzero]

    indice_image = 2 * g - r - b  # Calculate the 'ExG' index

    return indice_image


# ----------------------------------------------------------------------------------------------------------------------
# Workflow

start_time = time.time()

""

for image_path in input_folder.glob("*.png"):

    # Create a sub-folder for each of image (before/after cropping)
    image_name = image_path.stem
    image_output_folder = output_folder / image_name
    image_output_folder.mkdir(parents=True, exist_ok=True)
    image_output_folder2 = output_folder2 / image_name
    image_output_folder2.mkdir(parents=True, exist_ok=True)

    # read image: OpenCV uses the BGR color space by default, not the RGB space
    image = cv2.imread(str(image_path))  # RGB format input being converted to BGR format, discarding the alpha channel
    pil_image = Image.open(str(image_path))  # read the image with Pillow
    exif_data = pil_image.getexif()

    # Convert tag IDs to tag names
    if exif_data is not None:
        exif_data = {ExifTags.TAGS[k]: v for k, v in exif_data.items() if
                     k in ExifTags.TAGS and isinstance(ExifTags.TAGS[k], str)}

    # ··················································································································
    # Section 1: Segmentation, image to masks
    # ··················································································································
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)

    height, width, _ = image.shape  # get the image dimensions
    merged_mask = np.zeros((height, width), dtype=np.uint8)  # create an empty merged_mask in 2D array (0-255)
    full_mask = np.ones((height, width), dtype=np.uint8)

    for i, mask in enumerate(masks):
        mask_array = mask['segmentation']
        merged_mask[mask_array == 1] = i + 1  # add the current mask to the merged mask
        # save the masks
        cv2.imwrite(str(image_output_folder / f"mask_{i + 1}_{image_name}.png"), (mask_array * 255).astype(np.uint8))

    background_mask = full_mask - np.clip(merged_mask, 0, 1)
    cv2.imwrite(str(image_output_folder / f"background_mask_{image_name}.png"),
                (background_mask * 255).astype(np.uint8))

    metadata_header = "id,label,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h,mean_color_b,mean_color_g,mean_color_r,ExG,ExGR,GLI,GCC,GRVI,IKAW,MGRVI,MVARI,RGBVI,TGI,VARI,VDVI,NGRDI"
    metadata_records = []

    metadata_records = [
        [
            i,
            0,
            mask['area'],
            *mask['bbox'],
            *mask['point_coords'][0],
            mask['predicted_iou'],
            mask['stability_score'],
            *mask['crop_box'],
        ]
        for i, mask in enumerate(masks)
    ]

    metadata = [metadata_header]
    metadata.extend(','.join(map(str, record)) for record in metadata_records)
    with open(image_output_folder / f"metadata_{image_name}.csv", "w") as f:
        f.write("\n".join(metadata))

    # save merged mask with colors
    merged_mask_color = np.zeros((height, width, 3), dtype=np.uint8)  # create an empty merged_mask in 3D array (RGB color)
    color_list = generate_colors(100)
    for i, color in enumerate(color_list, start=1):
        merged_mask_color[merged_mask == i] = color

    # because cv2.imwrite will convert the color space,
    # so if you want yo correctly display the color bar, you need to convert it manually or using cv2.cvtColor
    merged_mask_color2 = cv2.cvtColor(merged_mask_color, cv2.COLOR_BGR2RGB)  # just for dealing with the conversion of cv2.imwrite
    cv2.imwrite(str(image_output_folder / f"mask_all_{image_name}.png"), merged_mask_color2)
    # ··················································································································

    # ··················································································································
    # Section 2: Cropping, crop the images using the mask
    # ··················································································································
    merged_cropped_img = np.zeros((height, width, 4), dtype=np.uint8)
    combined_mask = np.zeros((height, width), dtype=np.uint8)  # storge a combined mask for all the masks of this image
    merged_cropped_indice_image = []
    original_crop = []
    original_metadata_records = []

    for i, mask in enumerate(masks):
        mask_array = mask['segmentation'].astype(np.uint8)  # 2D mask array for every mask

        cropped_img = cv2.bitwise_and(image, image, mask=mask_array)  # crop the image, but in BGRA format

        # calculate the mean RGB and derivative Indices
        mean_color_b, mean_color_g, mean_color_r, _ = cv2.mean(cropped_img, mask=mask_array)  # cv2.mean() return the same order as the color channels of the input image.
        indices_crop = calculate_indices(mean_color_b, mean_color_g, mean_color_r)

        merged_cropped_indice_image.append(indices_crop['GLI'])  # add one indice to the merged_cropped_indice_image

        # add mean color per channel and BGR-based Indices to metadata
        metadata_record = metadata_records[i] + [mean_color_b, mean_color_g, mean_color_r] + list(indices_crop.values())  # create metadata record
        metadata_records[i] = metadata_record  # Update the record in metadata_records

        cropped_img_2 = np.zeros((height, width, 4), dtype=np.uint8)  # Initialize a 4-channel cropped image with zeros.
        cropped_img_2[:, :, :3] = cropped_img  # Assign the first three channels of masked_img_rgba with BGR masked_img.
        cropped_img_2[:, :, 3] = mask_array * 255  # Assign the fourth channel (alpha) with 255, which means the mask area is fully opaque.

        cv2.imwrite(str(image_output_folder2 / f"cropped_{i + 1}_{image_name}.png"), cropped_img_2)
        merged_cropped_img[mask_array == 1] = cropped_img_2[mask_array == 1]
        combined_mask = cv2.bitwise_or(combined_mask, mask_array)

        # pass the original crop
        original_crop.append(cropped_img_2)
        original_metadata_records.append(metadata_record)

    # save merged image
    merged_cropped_img[:, :, 3] = combined_mask * 255 # Set the alpha channel to 255 for the combined mask
    cv2.imwrite(str(image_output_folder2 / f"merged_{image_name}.png"), merged_cropped_img)

    # merged_cropped_indice_image = np.array(merged_cropped_indice_image)
    # merged_cropped_indices_image_output = np.zeros((height, width), dtype=np.float32)
    # for i, mask in enumerate(masks):
    #     mask_array = mask['segmentation'].astype(np.uint8)
    #     merged_cropped_indices_image_output[mask_array == 1] = merged_cropped_indice_image[i]
    #
    # imwrite(str(image_output_folder2 / f"merged_indice_{image_name}.tiff"), merged_cropped_indices_image_output)

    # Write the metadata to the CSV file
    metadata = [metadata_header]
    metadata.extend(','.join(map(str, record)) for record in metadata_records)
    with open(image_output_folder2 / f"metadata_{image_name}.csv", "w") as f:
        f.write("\n".join(metadata))

    # Save variables to a file for the next script
    with open(image_output_folder2 / 'variables.pkl', 'wb') as f:
        pickle.dump([original_crop, original_metadata_records, background_mask], f)



end_time = time.time()
elapsed_time_minutes = (time.time() - start_time) / 60
print(f"Elapsed time: {elapsed_time_minutes} seconds")

