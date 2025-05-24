import numpy as np

# helper function to cut off the parts of the image not in the mask
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

def fSNOWFLAKE_extractor(image, mask, threshold= 735) -> float:
    """Given an (RGB) image and associated binary mask, checks if there are
    white pixels in the region of interest. Returns 1 if that is the case,
    0 otherwise.
    
    :param image: The RGB image to be analyzed.
    :param mask: The binary mask of the image.
    :threshold: Threshold value for the evaluation of the image.
                Images with values lower than this parameter will
                return 0. Defaulted to 735.
    
    :return: Binary value 0 or 1, indicating presence of white pixels.
                
    """
    #checks if the image has a white-ish color
    cutten_img=cut_im_by_mask(image,mask)
    flat_im = np.reshape(cutten_img, (-1, 3))
    sum_img = np.sum(flat_im, axis=1)
    for x in sum_img:
        if x>threshold:
            return 1
    return 0