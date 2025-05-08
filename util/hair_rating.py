import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.metrics import accuracy_score
import pandas as pd
import os

import warnings
warnings.catch_warnings 

# the thresholds have been calibrated based on roughly 200 manually annotated hair images, and
# the code can be found in [rH_tuning_n_graphs.ipynb] 

def rate_hair(image, dst= 160, t1= 0.02, t2= 0.118, blur= True):
    """Function that, given an RGB image, extracts the number of pixels that constitute hair,
    computes the ratio of hair to total pixels, then assigns it a label between 0 (virtually no hair),
    1 (moderate amount of hair), and 2 (substantial amount of hair), based on the t1 and t2 thresholds.
    Returns both the ratio and the label.
    
    :param image: The RGB image to be analyzed.
    :param dst: Parameter value that indicates the maximum pixel value considered by the function,
                every instance of a brighter pixel will not be considered hair.
    :param t1: Lower threshold: all ratios smaller than t1 will be labelled as 0.
    :param t2: Higher threshold: ratios between t1 and t2 will be labeled as 1, higher values as 2.
    :param blur: Boolean value, if set to True, the image will be blurred before being analyzed.
    
    :return ratio, label, Mask:
    
    """
    
    image_size = image.shape[:2]
    img = image.mean(-1)

    if blur:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # -------------------------------------------------------- The Edges

    kernel = np.ones((3,3),np.uint8)
    img_filt = cv2.morphologyEx(np.uint8(img), cv2.MORPH_BLACKHAT, kernel) 
    img_filt = np.where(img_filt > 15, img_filt, 0)
    
    kernel = np.ones((4,4),np.uint8)
    img_filt = cv2.morphologyEx(img_filt, cv2.MORPH_DILATE, kernel)
        
    # -------------------------------------------------------- Edges within dark spots of image
        
    dark_spots = (img < dst).astype(np.uint8)
    kernel = np.ones((4,4),np.uint8)
    dark_spots = cv2.morphologyEx(dark_spots, cv2.MORPH_DILATE, kernel)
    
    img_filt = img_filt * dark_spots
    
    # -------------------------------------------------------- The Lines detected from the Edges
        
    lines = cv2.HoughLinesP(img_filt, cv2.HOUGH_PROBABILISTIC, np.pi / 90, 20, None, 1, 20)

    if lines is not None:
        lines = lines.reshape(-1, 4)
        N_lines = lines.shape[0]

        # exclude short lines
        lines_to_interp = []
        for ind in range(N_lines):
            line = lines[ind, :]
            x, y = fill_line(line[0::2], line[1::2], 1)
            lines_to_interp.append( (x, y) )
            
    else:
        lines_to_interp = []
        img_filt = np.zeros(image_size)

                
    # -------------------------------------------------------- The Final mask (from only reasonably longer lines)
    Mask = np.zeros_like(img_filt)
    for (x, y) in lines_to_interp:
        Mask[y, x] = 1

    kernel = np.ones((3,3),np.uint8)
    Mask = cv2.morphologyEx(Mask, cv2.MORPH_DILATE, kernel)
    Mask = Mask.astype(float)  

    # -------- Check if it is patchy enough (otherwise it's false positives; because hair is likely patchy)
    i, j = np.where( Mask != 0 )

    if i.size == 0:
        Mask = np.zeros(image_size)

    ratio = np.count_nonzero(Mask) / (image_size[0] * image_size[1])

    if ratio >= t2:
        label = 2
    elif ratio >= t1:
        label = 1
    else:
        label = 0
              
    return ratio, label, Mask

def fill_line(x, y, step=1):
    points = []
    if x[0] == x[1]:
        ys = np.arange(y.min(), y.max(), step)
        xs = np.repeat(x[0], ys.size)
    else:
        m = (y[1] - y[0]) / (x[1] - x[0])
        xs = np.arange(x[0], x[1], step * np.sign(x[1]-x[0]))
        ys = y[0] + m * (xs-x[0])
    return xs.astype(int), ys.astype(int)