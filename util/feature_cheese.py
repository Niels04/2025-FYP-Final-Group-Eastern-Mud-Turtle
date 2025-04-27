import numpy as np
from skimage.measure import label
from scipy.ndimage import binary_dilation, generate_binary_structure

def fCHEESE_extractor(mask) -> int:
    """Counts the number of connected components in the given binary mask.
    
    :param mask: The binary mask to be analyzed
    :return: The number of connected components in the mask
    
    """

    # create cross-shaped structure to dilate
    structure = generate_binary_structure(mask.ndim, 1)

    # dilate mask to close small gaps
    cc_list = []

    # dilate an increasing amount of times, save n. of cc for each iteration.
    # necessary since some masks do not behave as expected (potentially invisible
    # gaps make the number of cc skyroket, this prevents that, or at least provides
    # a more accurate estimate)
    for i in range(5, 26, 5):
        dilated_mask = binary_dilation(mask, structure= structure, iterations= i)

        # label mask
        labeled_mask = label(dilated_mask, connectivity= 2)

        # find number of cc: label() assigns an integer to
        # each cc, so the max value will be the total n of ccs.
        cc = labeled_mask.max()
        
        # append result
        cc_list.append(cc)
    
    # return mean cc number
    return sum(cc_list) // len(cc_list)
