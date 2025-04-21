import cv2
import numpy as np
from skimage.transform import resize
from sklearn.cluster import KMeans

import os
import random
import cv2

# cv2.setLogLevel(0)

# def readImageFile(file_path, is_mask = False):
#     if is_mask:
        
#         mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
#         return mask
    
#     img_bgr = cv2.imread(file_path)
#     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#     img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

#     return img_rgb, img_gray




# imagepath = r"Projects in Data Science\2025-FYP-Final-GroupE\data\MaskImagePair\PAT_76_1039_269.png"
# maskpath = r"Projects in Data Science\2025-FYP-Final-GroupE\data\MaskImagePair\PAT_76_1039_269_mask.png"

# img, imgbw = readImageFile(imagepath)
# mask = readImageFile(maskpath, is_mask=True)


def get_multicolor_rate(im, mask, n):

    # small tweak to ensure same image and mask shape
    target_shape = (mask.shape[0] // 4, mask.shape[1] // 4)
    im_rsz = cv2.resize(im, (target_shape[1], target_shape[0]))
    mask_rsz = cv2.resize(mask, (target_shape[1], target_shape[0]))
    im2 = im_rsz.copy()
    im2[mask_rsz == 0] = 0

    columns = im_rsz.shape[0]
    rows = im_rsz.shape[1]
    col_list = []
    for i in range(columns):
        for j in range(rows):
            if mask_rsz[i][j] != 0:
                col_list.append(im2[i][j] * 255) # instead of 256
    if len(col_list) == 0:
        return ""
    
    cluster = KMeans(n_clusters=n, n_init=10).fit(col_list)
    com_col_list = get_com_col(cluster, cluster.cluster_centers_)
    return com_col_list



def get_com_col(cluster, centroids):
    com_col_list = []
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins=labels)
    hist = hist.astype("float")
    hist /= hist.sum()

    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)], key= lambda x:x[0])
    for percent, color in colors:
        if percent > 0.08:
            com_col_list.append(color)
    return com_col_list


def fC_extractor(img, mask, n= 6, threshold = 30):
    color_list = get_multicolor_rate(img, mask, n)
    if not color_list:
        return []
    
    reduced_list = []

    for color in color_list:
        diff = True
        for unique_color in reduced_list:
            dist = np.linalg.norm(np.array(color) - np.array(unique_color))
            if dist < threshold:
                diff = False
                break
        if diff:
            reduced_list.append(color)
    return len(reduced_list)

#print(fC_extractor(img, mask))