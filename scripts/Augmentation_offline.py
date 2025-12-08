import os
import glob
import random
import shutil
from PIL import Image
import torchvision.transforms.functional as TF

def copy_and_augment_image_and_mask(image_path, mask_path, base_save_dir, num_copies=5):
    # Create directories for augmented images and masks if they do not exist
    image_save_dir = os.path.join(base_save_dir, 'Sample_')
    mask_save_dir = os.path.join(base_save_dir, 'Label_')
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)

    # Copy the original image and mask
    base_image_name = os.path.splitext(os.path.basename(image_path))[0]
    base_mask_name = os.path.splitext(os.path.basename(mask_path))[0]
    shutil.copy(image_path, os.path.join(image_save_dir, f'{base_image_name}.png'))
    shutil.copy(mask_path, os.path.join(mask_save_dir, f'{base_mask_name}.png'))

    # Load the original image and mask
    image = Image.open(image_path)
    mask = Image.open(mask_path)

    # Augmentation loop
    for i in range(num_copies):
        aug_image = image
        aug_mask = mask

        # Apply horizontal flip with 50% probability
        if random.random() < 0.5:
            aug_image = TF.hflip(image)
            aug_mask = TF.hflip(mask)

        # Apply vertical flip with 50% probability
        if random.random() < 0.5:
            aug_image = TF.vflip(image)
            aug_mask = TF.vflip(mask)

        # Apply random rotation between -10 and 10 degrees
        angle = random.uniform(-10, 10)
        aug_image = TF.rotate(aug_image, angle)
        aug_mask = TF.rotate(aug_mask, angle)

        # Save the augmented image and mask
        new_image_path = os.path.join(image_save_dir, f'{base_image_name}_aug_{i}.png')
        new_mask_path = os.path.join(mask_save_dir, f'{base_mask_name}_aug_{i}.png')
        aug_image.save(new_image_path)
        aug_mask.save(new_mask_path)


num_original_copies = 1  # Set to 1 to copy the original dataset without augmentation
num_augmented_copies = 3  # Set how many augmented copies you want

base_dir = 'G:/Project'
base_save_dir = 'G:/Project'
image_dir = os.path.join(base_dir, 'Sample_')
mask_dir = os.path.join(base_dir, 'Label_')
os.makedirs(image_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

image_paths = glob.glob(os.path.join(image_dir, '*.png'))
mask_paths = glob.glob(os.path.join(mask_dir, '*.png'))

# Ensure that image_paths and mask_paths are matched correctly
image_paths.sort()
mask_paths.sort()

for image_path, mask_path in zip(image_paths, mask_paths):
    copy_and_augment_image_and_mask(image_path, mask_path, base_save_dir, num_original_copies + num_augmented_copies - 1)
