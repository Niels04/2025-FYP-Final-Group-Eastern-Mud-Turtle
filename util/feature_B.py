import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import cv2
from skimage import measure
from sklearn.cluster import KMeans
from math import pi


def crop(mask):
        """Crops a given binary mask tighly.

        :param mask: Binary Mask to be cropped.
        :return: cropped mask"""
        #get y & x coordinates of nonzero points
        y_nonzero, x_nonzero = np.nonzero(mask)
        #set ylims to be lowest and highest points where mask contains nonzero pixel
        y_lims = [np.min(y_nonzero), np.max(y_nonzero)]
        #compute initial x limit
        x_lims = np.array([np.min(x_nonzero), np.max(x_nonzero)])
        #crop mask with computed limits & return
        return mask[y_lims[0]:y_lims[1], x_lims[0]:x_lims[1]]

def fB_extractor(mask):
    """
    Extracts feature B(\"Border Irregularity\") which is simply
    the compactness measure of the mask of the image.

    :param img: not used for now
    :param mask: binary mask of the image in cv2.GRAYSCALE format
    :return: Compactness measure from 0(not compact at all) to
    1(super compact, perfect circle)
    """
    cropped = crop(mask)
    padded = np.pad(cropped, pad_width=10, mode="constant", constant_values=0)#add padding so the masks don't touch the border

    #scale image such that the longer side is max. 512 pixels long -> differently sized images seem to yield different results
    h, w = padded.shape[:2]
    scaleFactor = 512 / max(w, h)
    scaled = cv2.resize(padded, (int(w*scaleFactor), int(h*scaleFactor)), interpolation=cv2.INTER_AREA)
    scaled = scaled.astype(bool)

    #calculate perimeter
    A = np.sum(scaled)
    perimeter = measure.perimeter_crofton(scaled, directions=4)
    if perimeter == 0:
        return 0.0

    #calculate and return final compactness measure
    compactness = (4*pi*A)/(perimeter**2)

    return compactness

#______THE FOLLOWING FUNCTIONS ARE ALL FOR THE FEATURE B FOR THE FORMULA (OPEN QUESTION)_____________

def find_midpoint_v4(mask):
        """Finds horizontal and vertical midpoints of the given binary
        mask, such that 50% of nonzero pixels in the mask
        are on either side of midpoint.

        :param mask: Binary Mask to be examined.
        :return x, y: coordinates of midpoint."""
        mX = 0
        mY = 0
        #get horizontal vector which contains number of nonzero pixels for each mask column
        summedX = np.sum(mask, axis=0)
        #calculate 50% of nonzero pixels in mask as threshold
        half_sumX = np.sum(summedX) / 2
        #iterate through columns until half of nonzero pixels in mask have been reached
        for i, n in enumerate(np.add.accumulate(summedX)):
            if n > half_sumX:
                #x-coordinate at which 50% of nonzero pixels where exceeded
                mX = i
                break
        
        summedY = np.sum(mask, axis=1)
        #calculate 50% of nonzero pixels in mask as threshold
        half_sumY = np.sum(summedY) / 2
        #iterate through rows until half of nonzero pixels in mask have been reached
        for i, n in enumerate(np.add.accumulate(summedY)):
            if n > half_sumY:
                # y-coordinate at which 50% of nonzero pixels where exceeded
                mY = i
                break

        return mX, mY

def cut_mask(mask):
    
    # input is numpy array mask
    col_sums = np.sum(mask, axis=0)     # sums up the values between 0 and 1
    row_sums = np.sum(mask, axis=1)     # shows if any row or column contains anything but 0s

    active_cols = []        # lists all the columns where there is no lesion
    for index, col_sum in enumerate(col_sums):  # takes all columns
        if col_sum != 0:                        # if the full column is 0, it's not added to the list
            active_cols.append(index)

    active_rows = []        # analog for rows
    for index, row_sum in enumerate(row_sums):
        if row_sum != 0:
            active_rows.append(index)

    # taking the bordering rows and columns of the mask (excluding the black edges where there is nothing)
    col_min = active_cols[0]
    col_max = active_cols[-1]
    row_min = active_rows[0]
    row_max = active_rows[-1]

    # saving the new mask
    cut_mask_ = mask[row_min:row_max+1, col_min:col_max+1]

    return cut_mask_

def cut_im_by_mask(image, mask):
    # same as previous function
    col_sums = np.sum(mask, axis=0)
    row_sums = np.sum(mask, axis=1)

    active_cols = []
    for index, col_sum in enumerate(col_sums):
        if col_sum != 0:
            active_cols.append(index)

    active_rows = []
    for index, row_sum in enumerate(row_sums):
        if row_sum != 0:
            active_rows.append(index)

    col_min = active_cols[0]
    col_max = active_cols[-1]
    row_min = active_rows[0]
    row_max = active_rows[-1]

    #except the cutting is applied to the image itself and not the mask
    cut_image = image[row_min:row_max+1, col_min:col_max+1]

    return cut_image

def sample_along_line_until_border(image, mask, center, angle_rad, step=1.0):
    """Sample pixel intensities along a radial line from the center until the lesion border is reached.\n
    :param image: Grayscale image
    :param mask: Binary lesion mask
    :param center: center of lesion as tuple (x, y)
    :param angle_rad: Angle (in radians) to sample along.
    :param step: Step size (in pixels) along the line.
    :return: 1D array of sampled intensities along the line.
    """
    cx, cy = center
    sampled_values = []
    
    zeroCount = 0
    d = 0
    while True:
        #get x & y coordinates for pixel at current distance
        x = cx + d * np.cos(angle_rad)
        y = cy + d * np.sin(angle_rad)
        d+=1

        # Clip to image boundaries
        if x < 0 or x >= mask.shape[0] or y < 0 or y >= mask.shape[1]:
            break

        # Use bilinear interpolation to sample intensity
        px = cv2.getRectSubPix(image, (1, 1), (x, y))[0, 0]
        mx = cv2.getRectSubPix(mask.astype(np.uint8), (1, 1), (x, y))[0, 0]
        if mx == 0:
            zeroCount += 1
        if zeroCount > 20:
            break#reached the outside of the mask

        sampled_values.append(px)

    return np.array(sampled_values)

def analyze_sector_gradients(image, mask, center, angle_start, angle_end, lines_per_sector=10):
    """For the sector in the given image masked with the given mask, return the mean of the maximum gradients
    encountered along 10 sampled radial lines insdie the sector between angle_start & angle_end
    :param image: image to analyze
    :param mask: mask to apply to image before analysis
    :param center: center from which to measure the sector
    :param angle_start: starting angle of sector in degrees
    :param angle_end: ending angle of sector in degrees
    :param lines_per_sector: number of lines to be sampled in this sector
    :return: mean of the maximum gradients across all lines in this sector
    """
    start_rad = np.radians(angle_start)
    end_rad = np.radians(angle_end)
    angle_range = np.linspace(start_rad, end_rad, lines_per_sector)#get angles for lines along which to sample
    center_cv = (center[1], center[0])
    max_gradients = []

    for angle in angle_range:#iterate through angles
        intensities = sample_along_line_until_border(image, mask, center_cv, angle)#sample along line at current angle
        gradients = np.abs(np.gradient(intensities))#get 1d gradient from the line samples
        max_gradients.append(np.max(gradients))#add max gradient along this line

    return np.mean(max_gradients)

def fB_formula(img, mask, nSectors=8):
    """Extract the \"irregular Boder\" feature,
    which is a number from 0 to 1 that is a measure
    for the difference between the center intensity
    and border intensity of the lesion.
    
    :param img: the image to process
    :param mask: mask to apply to the image
    :return: border irregularity measure from
    0(regular) to 1(irregular)"""
    #Preprocess: Apply a gaussean blur
    img = cv2.GaussianBlur(img, ksize=(3, 3), sigmaX=0)

    cutImg = cut_im_by_mask(img, mask)
    cutImgGray = cv2.cvtColor(cutImg, cv2.COLOR_RGB2GRAY)#convert image to grayscale for gradient analysis
    cutMask = cut_mask(mask)
    mX, mY = find_midpoint_v4(cutMask)

    #store max gradient values for the sectors
    gradScores = []

    sectorDeg = int(np.ceil(360 / nSectors))
    for deg in range(0, 360, sectorDeg):
        #analyze gradient in this sector
        avgMaxGrad = analyze_sector_gradients(cutImgGray, cutMask, (mX, mY), deg, deg+sectorDeg)

        gradScores.append(avgMaxGrad)

    gradScores = np.array(gradScores)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(gradScores.reshape(-1, 1))
    # Get cluster centers (means)
    centers = kmeans.cluster_centers_.flatten()  # Shape (2,)
    # Identify the cluster with the lower mean
    lower_mean_cluster = np.argmax(centers)
    # Get labels
    labels = kmeans.labels_
    # Count sectors assigned to the lower-mean cluster
    count = np.sum(labels == lower_mean_cluster)
    return count

