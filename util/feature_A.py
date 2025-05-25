from PIL import Image
import numpy as np 
from math import sqrt, floor, ceil, nan, pi
from skimage.transform import rotate

#midpoint finder function

def find_midpoint_v1(image):
    row_mid = image.shape[0] / 2     # nr of the middle row
    col_mid = image.shape[1] / 2     # nr of the middle column
    return row_mid, col_mid

    #asymmetry score function

def asymmetry(mask):
    
    row_mid, col_mid = find_midpoint_v1(mask)           # uses the previous function

    # slicing into 4 different halves. We will calculate symmetry on both halves
    # ceil and floor are used in case the outputs of the midpoint functions have decimals
    upper_half = mask[:ceil(row_mid), :]
    lower_half = mask[floor(row_mid):, :]
    left_half = mask[:, :ceil(col_mid)]
    right_half = mask[:, floor(col_mid):]

    # flip one of the complementary halves of each pair (one horizontal half, one vertical half)
    flipped_lower = np.flip(lower_half, axis=0)
    flipped_right = np.flip(right_half, axis=1)

    # using the xor logic, makes a binary array of all pixels in both symmetry pairs
    hori_xor_area = np.logical_xor(upper_half, flipped_lower)
    vert_xor_area = np.logical_xor(left_half, flipped_right)

    # to calculate ratio, we need to know how many pixels there are in total
    total_pxls = np.sum(mask)
    # the sum together how many true values there are
    # keeping note of xor logic => the more ASYMMETRY the higher the number
    hori_asymmetry_pxls = np.sum(hori_xor_area)
    vert_asymmetry_pxls = np.sum(vert_xor_area)

    asymmetry_score = (hori_asymmetry_pxls + vert_asymmetry_pxls) / (total_pxls * 2)

    return round(asymmetry_score, 4)

    
#crops the mask to just the lesion

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

#rotates the mask n times and calculate the asymmetry from each angle

def rotation_asymmetry(mask, n:int = 30):

    asymmetry_scores = {}

    for i in range(n):

        degrees = 90 * i / n

        rotated_mask = rotate(mask, degrees,preserve_range=True)
        rotated_mask = rotated_mask > 0.5 #insure that it is binarized
        cutted_mask = cut_mask(rotated_mask)

        asymmetry_scores[degrees] = asymmetry(cutted_mask)

    return asymmetry_scores

# image is not accessed
def fA_extractor(mask): #takes in a file path to image and mask
    """Given a grayscale mask, computes and returns both mean asymmetry score
    and worst asymmetry score.
    
    :param mask: The grayscale mask to be analyzed.
    
    :return: mean asymmetry score, worst asymmetry score."""
    
    #calculates the mean from the rotation asymmetry
    asymmetry_scores = rotation_asymmetry(mask) 
    worst_score = max(asymmetry_scores.values())
    mean_score = sum(asymmetry_scores.values()) / len(asymmetry_scores)

    return mean_score, worst_score
    
    
# TEMPORARY
def fA_formula(mean_score, worst_score):
    form_mean = mean_score * 2
    form_worst = worst_score * 2

    return form_mean, form_worst
    
def fA_formula_v2(mask):

    def asymmetry_v2(mask):
        row_mid, col_mid = find_midpoint_v1(mask)           # uses the previous function

        asymmetry_score = 2

        # slicing into 4 different halves. We will calculate symmetry on both halves
        # ceil and floor are used in case the outputs of the midpoint functions have decimals
        upper_half = mask[:ceil(row_mid), :]
        lower_half = mask[floor(row_mid):, :]
        left_half = mask[:, :ceil(col_mid)]
        right_half = mask[:, floor(col_mid):]

        # flip one of the complementary halves of each pair (one horizontal half, one vertical half)
        flipped_lower = np.flip(lower_half, axis=0)
        flipped_right = np.flip(right_half, axis=1)

        if flipped_lower.all() == np.array(upper_half).all():
            asymmetry_score -= 1
        if flipped_right.all() == np.array(left_half).all():
            asymmetry_score -= 1

        return asymmetry_score
    
    cutted_mask = cut_mask(mask)
    return asymmetry_v2(cutted_mask)

def fA_formula_v3(mask):

    def asymmetry_v3(mask):
        row_mid, col_mid = find_midpoint_v1(mask)           # uses the previous function

        asymmetry_score = 2

        # slicing into 4 different halves. We will calculate symmetry on both halves
        # ceil and floor are used in case the outputs of the midpoint functions have decimals
        upper_half = mask[:ceil(row_mid), :]
        lower_half = mask[floor(row_mid):, :]
        left_half = mask[:, :ceil(col_mid)]
        right_half = mask[:, floor(col_mid):]

        # flip one of the complementary halves of each pair (one horizontal half, one vertical half)
        flipped_lower = np.flip(lower_half, axis=0)
        flipped_right = np.flip(right_half, axis=1)

            # using the xor logic, makes a binary array of all pixels in both symmetry pairs
        hori_xor_area = np.logical_xor(upper_half, flipped_lower)
        vert_xor_area = np.logical_xor(left_half, flipped_right)

        # to calculate ratio, we need to know how many pixels there are in total
        total_pxls = np.sum(mask)
        # the sum together how many true values there are
        # keeping note of xor logic => the more ASYMMETRY the higher the number
        hori_asymmetry_pxls = np.sum(hori_xor_area)
        vert_asymmetry_pxls = np.sum(vert_xor_area)

        if flipped_lower.all() == np.array(upper_half).all():
            asymmetry_score -= 1
        if flipped_right.all() == np.array(left_half).all():
            asymmetry_score -= 1

        return asymmetry_score
    
    cutted_mask = cut_mask(mask)
    return asymmetry_v3(cutted_mask)
    



        