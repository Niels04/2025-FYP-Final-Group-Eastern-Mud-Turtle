import numpy as np
import cv2
from skimage import measure
from math import pi

def crop(mask):
        """Crops a given binary mask tighlyr.

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