# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
import glob


img_h, img_w = 512, 512  # Specify the desired image height and width
sums, sq_sums, count = np.zeros(3), np.zeros(3), 0

imgs_path = r'/path'
imgs_path_list = glob.glob(os.path.join(imgs_path, 'image_*.png'))


for item in imgs_path_list:
    img = cv2.imread(item)
    img = cv2.resize(img, (img_w, img_h))
    # Convert to float32 on a per-image basis to save memory
    img = img.astype(np.float32) / 255.

    # Create mask where pixels are not zero (non-background)
    non_background_mask = img > 0

    # Count non-background pixels
    non_background_count = np.sum(non_background_mask, axis=(0, 1))
    count += non_background_count

    # Calculate sum and sum of squares for non-background pixels
    sums += np.sum(img * non_background_mask, axis=(0, 1))
    sq_sums += np.sum((img * non_background_mask) ** 2, axis=(0, 1))

# Calculate means and standard deviations
means = sums / count
variances = (sq_sums / count) - (means ** 2)
stdevs = np.sqrt(variances)

# BGR to RGB if necessary (since OpenCV reads images in BGR)
means = means[::-1]
stdevs = stdevs[::-1]

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
