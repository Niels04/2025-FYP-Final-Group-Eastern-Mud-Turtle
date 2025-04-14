import numpy as np
import skimage.measure
from math import pi

def fB_extractor(img, mask):
    """
    Extracts feature B(\"Border Irregularity\") which is simply
    the compactness measure of the mask of the image.

    :param img: not used for now
    :param mask: binary mask of the image in cv2.GRAYSCALE format
    :return: Compactness measure from 0(not compact at all) to
    1(super compact, perfect circle)
    """
    mask = mask.astype(bool)
    A = np.sum(mask)
    perimeter = measure.perimeter(mask, neighborhood=8)
    if perimeter == 0:
        return 0.0

    compactness = (4*pi*A)/(perimeter**2)
    score = round(1-compactness, 3)

    return compactness