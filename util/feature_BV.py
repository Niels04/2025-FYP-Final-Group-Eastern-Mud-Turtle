# import necessary libraries
import numpy as np
import cv2

def fBV_extractor(image, mask) -> float:
    """Computes the percentage of the lesion in the given (RGB) image
    that consists of blue / purple-ish pixels (blue veils).
    It outputs a value between 0 and 1, where a 0 indicates total 
    absence of blue veils, and a 1 indicates that the lesion is
    completely covered in blue veils.
    
    :param image: The RGB image to be analysed
    :param mask: The mask for the given image
    :return: Float with value between 0 and 1
    
    """

    # store n. of pixels in the mask
    tot_pixels = np.count_nonzero(mask)

    # convert image to HSV and apply mask
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # make the mask 3D
    mask_3D = mask[:, :, np.newaxis]

    # apply mask
    image_hsv = image_hsv * mask_3D

    # extract H channel
    h = image_hsv[:, :, 0]

    # create mask to filter out all pixels that are not blue / purple-ish
    bp_mask = (image_hsv[:, :, 0] >= 100) & (image_hsv[:, :, 0] <= 145)
    # Note: H is in degrees, but in cv2 its range is not 0-360 but 0-179, so we halve the values

    # store n. of blue / purple pixels
    veil_pixels = np.count_nonzero(bp_mask)

    # compute portion of lesion covered by blue veils
    BV = veil_pixels / tot_pixels
    # value from 0 to 1: from no blue veils to entirely covered in blue veils

    return BV