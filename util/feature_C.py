import cv2
import numpy as np
from skimage.transform import resize
from sklearn.cluster import KMeans

def fC_extractor(img, mask, n= 6, threshold = 30):
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
    
    def get_multicolor_rate(im, mask, n):
        im = resize(im, (im.shape[0] // 4, im.shape[1] // 4), anti_aliasing=True)
        mask = resize(
            mask, (mask.shape[0] // 4, mask.shape[1] // 4), anti_aliasing=True
        )
        im2 = im.copy()
        im2[mask == 0] = 0

        columns = im.shape[0]
        rows = im.shape[1]
        col_list = []
        for i in range(columns):
            for j in range(rows):
                if mask[i][j] != 0:
                    col_list.append(im2[i][j] * 256)
        if len(col_list) == 0:
            return ""
        
        cluster = KMeans(n_clusters=n, n_init=10).fit(col_list)
        com_col_list = get_com_col(cluster, cluster.cluster_centers_)
        return com_col_list
    
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

imagepath = r"Projects in Data Science\2025-FYP-Final-GroupE\data\MaskImagePair\PAT_76_1039_269.png"
maskpath = r"Projects in Data Science\2025-FYP-Final-GroupE\data\MaskImagePair\PAT_76_1039_269_mask.png"

image = cv2.imread("imagepath")
#image_2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
mask = cv2.imread("maskpath", 0)

if image is None:
    raise FileNotFoundError(f"Image not found: {imagepath}")
if mask is None:
    raise FileNotFoundError(f"Mask not found: {maskpath}")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

colors = fC_extractor(image_rgb, mask, n=8)
print(f"Found {colors} distinct colors.")