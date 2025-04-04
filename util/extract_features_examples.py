import cv2
import numpy as np
from math import sqrt, floor, ceil, nan, pi
from skimage import color, exposure
from skimage.color import rgb2gray
from skimage.feature import blob_log
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.transform import resize
from skimage.transform import rotate
from skimage import morphology
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from skimage.color import rgb2hsv
from scipy.stats import circmean, circvar, circstd
from statistics import variance, stdev
from scipy.spatial import ConvexHull


def measure_pigment_network(image) -> int:
    """Computes the percentage of the (BGR) image that constitutes the foreground.
    
    :param image: The BGR image to be analysed
    :return: The % of the foreground in the image
    
    """

    # convert image from BGR to LAB
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # extract lightness value from image,
    # cv2.split() divides the image into
    # its color channels
    l_channel, _, _ = cv2.split(lab_image)

    # increase contrast to better separate background and foreground
    enhanced_l_channel = cv2.equalizeHist(l_channel)

    # separate background (set pixels to 0) from
    # foreground (set pixels to 255). 
    # THRESH_OTSU computes the best threshold to do so
    _, binary_mask = cv2.threshold(enhanced_l_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # compute total number of pixels in the image
    total_pixels = np.prod(binary_mask.shape[:2])

    # count number of pixels in the lesion
    pigment_pixels = np.count_nonzero(binary_mask)

    # compute % of pixels in the lesion region
    coverage_percentage = (pigment_pixels / total_pixels) * 100

    return coverage_percentage


def measure_blue_veil(image) -> int:
    """Computes the number of blue-ish pixels in the given (BGR) image.
    
    :param image: The BGR image to be analysed
    :return: integer representing the number of blue-ish pixels
    
    """

    # extract image dimensions
    height, width, _ = image.shape

    # initialize counter for number of blue veils
    count = 0

    # iterate through all the pixels in the image
    for y in range(height):
        for x in range(width):

            # extract color values from image
            b, g, r = image[y, x]

            # update counter if the pixel is teal, blue, purple
            if b > 60 and (r - 46 < g) and (g < r + 15):
                count += 1

    return count


def measure_vascular(image) -> int:
    """Computes the number of pixels that identify blood or veins in the given (RGB) image.
    DO NOT USE (weird choices)

    :param image: The RGB image to be analysed
    :return: The number of pixels that identify blood or veins."""
    
    # extract the red channel of the image
    red_channel = image[:, :, 0]

    # enhance the red channel (doesn't do anything as of now)
    enhanced_red_channel = exposure.adjust_gamma(red_channel, gamma=1)

    # copy the image and apply the enhanced red channel
    enhanced_image = image.copy()
    enhanced_image[:, :, 0] = enhanced_red_channel

    # convert image to hsv
    hsv_img = color.rgb2hsv(enhanced_image)

    # set hsv thresholds to find pixels of certain colors
    lower_red1 = np.array([0, 40/100, 00/100])
    upper_red1 = np.array([25/360, 1, 1])
    mask1 = np.logical_and(np.all(hsv_img >= lower_red1, axis=-1), np.all(hsv_img <= upper_red1, axis=-1))

    # repeat, for different shades and tones
    lower_red2 = np.array([330/360, 40/100, 00/100])  
    upper_red2 = np.array([1, 1, 1]) 
    mask2 = np.logical_and(np.all(hsv_img >= lower_red2, axis=-1), np.all(hsv_img <= upper_red2, axis=-1))

    # combine the two masks
    mask = np.logical_or(mask1, mask2)

    # return the number of pixels present in the mask
    return np.sum(mask)


def measure_globules(image) -> int:
    """Computes the number of blobs in the given (RGB) image.
    
    :param image: The RGB image to be analysed
    :return: The number of blobs in the image

    """

    # convert to grayscale, then save the inverted image
    image_gray = rgb2gray(image)
    inverted_image = 1 - image_gray

    # find blobs, then estimate radius by multiplying associated sigma with sqrt(2)
    blobs_doh = blob_log(inverted_image, min_sigma=1, max_sigma=4, num_sigma=50, threshold=.05)
    blobs_doh[:, 2] = blobs_doh[:, 2] * sqrt(2)

    # compute number of blobs
    blob_amount = len(blobs_doh)

    return blob_amount

def measure_streaks(image) -> int:
    """Computes the irregularity of the lesion in the given (BGR) image.
    
    :param image: The BGR image to be analysed
    :return: The irregularity score of the lesion

    """
   # get grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # compute threshold, using 11 neighbors
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # find the contours of the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # compute area and perimeter of the contours
    lesion_area = cv2.contourArea(contours[0])
    border_perimeter = cv2.arcLength(contours[0], True)

    # finally evaluate irregularity
    if lesion_area == 0:
        irregularity = 0
    else:
        irregularity = (border_perimeter ** 2) / (4 * np.pi * lesion_area)

    return irregularity


def measure_irregular_pigmentation(image):
    """Requires BGR images. God knows what use could this mess have. DO NOT USE for now.
    
    :param image: The BGR image to be analysed(?)
    :return: Nothing useful, and definitely not what the name suggests."""
    
    # get grayscale image and find threshold. Apply mask
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold = threshold_otsu(gray)
    binary = gray > threshold

    # label connected foreground components
    labeled_image = label(binary)

    min_rows, min_cols, max_rows, max_cols = [], [], [], []

    # iterate through connected components
    for region in regionprops(labeled_image):

        # store area and perimeter
        area = region.area
        perimeter = region.perimeter

        if perimeter == 0:
            continue
        
        # compute circularity of the region
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        
        # save bbox sides if condition is met
        if circularity < 0.6:
            min_row, min_col, max_row, max_col = region.bbox
            min_rows.append(min_row)
            min_cols.append(min_col)
            max_rows.append(max_row)
            max_cols.append(max_col)

    # for some reason, compute the percentage of the image that is occupied by
    # the lesion and return absolutely nothing else, leaving zero trace of all
    # previous computations
    _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    total_pixels = np.prod(binary_mask.shape[:2])
    irregular_pixels = np.count_nonzero(binary_mask)
    coverage_percentage = (irregular_pixels / total_pixels) * 100

    return coverage_percentage


def measure_regression(image):
    """Computes number of bright, non-saturated pictures in the image.
    
    :param image: The BGR image to be analysed
    :return: The number of pixels in the image with
    low saturation (0-30) and high brightness(150-255)
    measured in the hsv color fomat"""
   
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #set lower and upper limit for hsv values of colors to be selected.
    lower_color = np.array([0, 0, 150])
    upper_color = np.array([180, 30, 255])
    #mask pixels in range
    mask = cv2.inRange(hsv_img, lower_color, upper_color)
    #get number of pixels in range
    num_pixels = cv2.countNonZero(mask)

    return num_pixels

def get_compactness(mask):
    """Computes the compactness measure for the given binary mask.
    
    :param mask: The Binary mask to be analyzed
    :return: Compactness measure \"Circularity\"
    from 0 to inf, where lower values indicate
    higher compactness."""
    # mask = color.rgb2gray(mask)
    area = np.sum(mask)
    struct_el = morphology.disk(3)
    #erode mask once
    mask_eroded = morphology.binary_erosion(mask, struct_el)
    #subtract eroded mask from original mask to obtain border pixels -> get number of border pixels with sum
    perimeter = np.sum(mask - mask_eroded)
    #calculate and return "compactness" measure
    return perimeter**2 / (4 * np.pi * area)

def get_asymmetry(mask):
    """Computes the asymmetry for a given binary mask
    by rotating 6 times 30°.
    
    :param mask: The Binary mask to be analyzed
    :return: Asymmetry measure from 0 to 1 where
    0 is perfect symmetry and 1 is perfect asymmetry."""
    # mask = color.rgb2gray(mask)
    scores = []
    for _ in range(6):
        #crop mask to only look at the nonzero area
        segment = crop(mask)
        #(np.sum(segment))
        #append ratio of non-overlapping pixels between flipped mask and original mask over total mask pixels
        scores.append(np.sum(np.logical_xor(segment, np.flip(segment))) / (np.sum(segment)))
        #rotate mask for next iteration
        mask = rotate(mask, 30)
        #return average score
    return sum(scores) / len(scores)

def get_multicolor_rate(im, mask, n):
    """Computes the maximum euclidean distance between
    2 dominant colors in the masked area of the image
    that are adjacent in the hierarchy of dominance.
    Uses kmeans with n clusters to find centroids that
    represent \"dominant\" colors and compares only
    colors such that > 8% of the masked area are
    clustered with this color.
    
    :param image: The BFR or RGB image to analyze
    :param mask: The binary mask to apply to the image
    :param n: Number of clusters to use forMeans Color Clustering
    :return: Maxiumum euclidean distance between 2 dominant colors
    in the masked area of the image, which are adjacent in the
    hierarchy of dominance. Note that return is the same for BGR
    and RGB images."""
    # mask = color.rgb2gray(mask)
    #scale image down for performance
    im = resize(im, (im.shape[0] // 4, im.shape[1] // 4), anti_aliasing=True)
    mask = resize(
        mask, (mask.shape[0] // 4, mask.shape[1] // 4), anti_aliasing=True
    )
    im2 = im.copy()
    im2[mask == 0] = 0

    columns = im.shape[0]
    rows = im.shape[1]
    col_list = []
    #create list of all colors present in masked area of image
    for i in range(columns):
        for j in range(rows):
            if mask[i][j] != 0:
                col_list.append(im2[i][j] * 256)
    #return empty string if nothing masked
    if len(col_list) == 0:
        return ""
    #find n clusters of dominant colors
    cluster = KMeans(n_clusters=n, n_init=10).fit(col_list)
    #obtain sorted list of dominant colors that cover > 8% of masked area
    com_col_list = get_com_col(cluster, cluster.cluster_centers_)

    dist_list = []
    m = len(com_col_list)
    #return empty string if there is only 1 or 0 dominant colors
    if m <= 1:
        return ""
    #compute euclidean distances between adjacent colors
    for i in range(0, m - 1):
        j = i + 1
        col_1 = com_col_list[i]
        col_2 = com_col_list[j]
        dist_list.append(
            np.sqrt(
                (col_1[0] - col_2[0]) ** 2
                + (col_1[1] - col_2[1]) ** 2
                + (col_1[2] - col_2[2]) ** 2
            )
        )
    #return max distance between 2 adjacent colors
    return np.max(dist_list)

def get_com_col(cluster, centroids):
    """Computes a sorted list of dominant colors given
    the KMeans clusters and centroids for all clusters
    that contain > 8% of pixels. The list is sorted by
    the proportion of pixels covered by this cluster.
    
    :param cluster: KMeans object fitted on pixel values of an image.
    :param centroids: centroids of the KMeans algorithm, i.e. central
    pixels of the clusters specified in \"cluster\".
    :return: Sorted list of dominant colors, sorted by proportion,
    excluding all clusters that cover <8% of the pixels."""
    com_col_list = []
    #create labels for the number of clusters (dominant colors)
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    #count frequencies of each label with histogram
    (hist, _) = np.histogram(cluster.labels_, bins=labels)
    hist = hist.astype("float")
    #normalize histogram to calculate proportions
    hist /= hist.sum()

    #Commented out the parts that are redundant for computationand  and only produce a visualization
    #rect = np.zeros((50, 300, 3), dtype=np.uint8)
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)], key= lambda x:x[0])
    #start = 0
    for percent, color in colors:
        if percent > 0.08:
            com_col_list.append(color)
        # end = start + (percent * 300)
        # cv2.rectangle(
        #     rect,
        #     (int(start), 0),
        #     (int(end), 50),
        #     color.astype("uint8").tolist(),
        #     -1,
        # )
        # start = end
    return com_col_list

def crop(mask):
        """Crops a given binary mask tighly on the vertical axis
        and centrally on the horizontal axis. The x-axis center is
        measured such that on both sides there are 50%
        of nonzero mask pixels. The border for the x-crop
        is then the midpoing +- the larger distance from
        the midpoint to the left and right border.

        :param mask: Binary Mask to be cropped.
        :return: cropped mask"""
        #compute x-midpoint
        mid = find_midpoint_v4(mask)
        #get y & x coordinates of nonzero points
        y_nonzero, x_nonzero = np.nonzero(mask)
        #set ylims to be lowest and highest points where mask contains nonzero pixel
        y_lims = [np.min(y_nonzero), np.max(y_nonzero)]
        #compute initial x limit
        x_lims = np.array([np.min(x_nonzero), np.max(x_nonzero)])
        #compute x-limit from x-center +- larger distance from midpoint to left/right border
        x_dist = max(np.abs(x_lims - mid))
        x_lims = [mid - x_dist, mid+x_dist]
        #crop mask with computed limits & return
        return mask[y_lims[0]:y_lims[1], x_lims[0]:x_lims[1]]

def find_midpoint_v4(mask):
        """Finds horizontal midpoint of the given binary
        mask, such that 50% of nonzero pixels in the mask
        are on either side of midpoint.

        :param mask: Binary Mask to be examined.
        :return x: horizontal coordinate of midpoint."""
        #get horizontal vector which contains number of nonzero pixels for each mask column
        summed = np.sum(mask, axis=0)
        #calculate 50% of nonzero pixels in mask as threshold
        half_sum = np.sum(summed) / 2
        #iterate through columns until half of nonzero pixels in mask have been reached
        for i, n in enumerate(np.add.accumulate(summed)):
            if n > half_sum:
                #return x-coordinate at shich 50% of nonzero pixels where exceeded
                return i
  

def cut_mask(mask):
    
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

    cut_mask_ = mask[row_min:row_max+1, col_min:col_max+1]

    return cut_mask_

def cut_im_by_mask(image, mask):
    

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

    cut_image = image[row_min:row_max+1, col_min:col_max+1]

    return cut_image

def find_midpoint_v1(image):
    
    row_mid = image.shape[0] / 2
    col_mid = image.shape[1] / 2
    return row_mid, col_mid

def asymmetry(mask):
    

    row_mid, col_mid = find_midpoint_v1(mask)

    upper_half = mask[:ceil(row_mid), :]
    lower_half = mask[floor(row_mid):, :]
    left_half = mask[:, :ceil(col_mid)]
    right_half = mask[:, floor(col_mid):]

    flipped_lower = np.flip(lower_half, axis=0)
    flipped_right = np.flip(right_half, axis=1)

    hori_xor_area = np.logical_xor(upper_half, flipped_lower)
    vert_xor_area = np.logical_xor(left_half, flipped_right)

    total_pxls = np.sum(mask)
    hori_asymmetry_pxls = np.sum(hori_xor_area)
    vert_asymmetry_pxls = np.sum(vert_xor_area)

    asymmetry_score = (hori_asymmetry_pxls + vert_asymmetry_pxls) / (total_pxls * 2)

    return round(asymmetry_score, 4)

def rotation_asymmetry(mask, n: int):

    asymmetry_scores = {}

    for i in range(n):

        degrees = 90 * i / n

        rotated_mask = rotate(mask, degrees)
        cutted_mask = cut_mask(rotated_mask)

        asymmetry_scores[degrees] = asymmetry(cutted_mask)

    return asymmetry_scores

def mean_asymmetry(mask, rotations = 30):
    
    asymmetry_scores = rotation_asymmetry(mask, rotations)
    mean_score = sum(asymmetry_scores.values()) / len(asymmetry_scores)

    return mean_score          

def best_asymmetry(mask, rotations = 30):
    
    asymmetry_scores = rotation_asymmetry(mask, rotations)
    best_score = min(asymmetry_scores.values())

    return best_score

def worst_asymmetry(mask, rotations = 30):
    
    asymmetry_scores = rotation_asymmetry(mask, rotations)
    worst_score = max(asymmetry_scores.values())

    return worst_score  

def slic_segmentation(image, mask, n_segments = 50, compactness = 0.1):
    
    slic_segments = slic(image,
                    n_segments = n_segments,
                    compactness = compactness,
                    sigma = 1,
                    mask = mask,
                    start_label = 1,
                    channel_axis = 2)
    
    return slic_segments

def get_rgb_means(image, slic_segments):
    
    max_segment_id = np.unique(slic_segments)[-1]

    rgb_means = []
    for i in range(1, max_segment_id + 1):

        segment = image.copy()
        segment[slic_segments != i] = -1

        rgb_mean = np.mean(segment, axis = (0, 1), where = (segment != -1))
        
        rgb_means.append(rgb_mean) 
        
    return rgb_means

def get_hsv_means(image, slic_segments):
    
    hsv_image = rgb2hsv(image)

    max_segment_id = np.unique(slic_segments)[-1]

    hsv_means = []
    for i in range(1, max_segment_id + 1):

        segment = hsv_image.copy()
        segment[slic_segments != i] = nan

        hue_mean = circmean(segment[:, :, 0], high=1, low=0, nan_policy='omit') 
        sat_mean = np.mean(segment[:, :, 1], where = (slic_segments == i))  
        val_mean = np.mean(segment[:, :, 2], where = (slic_segments == i)) 

        hsv_mean = np.asarray([hue_mean, sat_mean, val_mean])

        hsv_means.append(hsv_mean)
        
    return hsv_means

def rgb_var(image, slic_segments):
    

    if len(np.unique(slic_segments)) == 2: 
        return 0, 0, 0

    rgb_means = get_rgb_means(image, slic_segments)
    n = len(rgb_means) 

    red = []
    green = []
    blue = []
    for rgb_mean in rgb_means:
        red.append(rgb_mean[0])
        green.append(rgb_mean[1])
        blue.append(rgb_mean[2])

    red_var = variance(red, sum(red)/n)
    green_var = variance(green, sum(green)/n)
    blue_var = variance(blue, sum(blue)/n)

    return red_var, green_var, blue_var

def hsv_var(image, slic_segments):
    
    if len(np.unique(slic_segments)) == 2: 
        return 0, 0, 0

    hsv_means = get_hsv_means(image, slic_segments)
    n = len(hsv_means) 

    hue = []
    sat = []
    val = []
    for hsv_mean in hsv_means:
        hue.append(hsv_mean[0])
        sat.append(hsv_mean[1])
        val.append(hsv_mean[2])

    hue_var = circvar(hue, high=1, low=0)
    sat_var = variance(sat, sum(sat)/n)
    val_var = variance(val, sum(val)/n)

    return hue_var, sat_var, val_var


def color_dominance(image, mask, clusters = 5, include_ratios = False):
    
    cut_im = cut_im_by_mask(image, mask) 
    hsv_im = rgb2hsv(cut_im) 
    flat_im = np.reshape(hsv_im, (-1, 3)) 

    k_means = KMeans(n_clusters=clusters, n_init=10, random_state=0)
    k_means.fit(flat_im)

    dom_colors = np.array(k_means.cluster_centers_, dtype='float32') 

    if include_ratios:

        counts = np.unique(k_means.labels_, return_counts=True)[1] 
        ratios = counts / flat_im.shape[0] 

        r_and_c = zip(ratios, dom_colors) 
        r_and_c = sorted(r_and_c, key=lambda x: x[0],reverse=True) 

        return r_and_c
    
    return dom_colors

def compactness_score(mask):
     
    A = np.sum(mask)

    struct_el = morphology.disk(2)

    mask_eroded = morphology.binary_erosion(mask, struct_el)

    perimeter = mask - mask_eroded

    l = np.sum(perimeter)

    compactness = (4*pi*A)/(l**2)

    score = round(1-compactness, 3)

    return compactness

def convexity_score(mask):

    coords = np.transpose(np.nonzero(mask))

    hull = ConvexHull(coords)

    lesion_area = np.count_nonzero(mask)

    convex_hull_area = hull.volume + hull.area

    convexity = lesion_area / convex_hull_area
    
    return convexity 

def get_relative_rgb_means(image, slic_segments):

    max_segment_id = np.unique(slic_segments)[-1]

    rgb_means = []
    for i in range(0, max_segment_id + 1):

        segment = image.copy()
        segment[slic_segments != i] = -1

        rgb_mean = np.mean(segment, axis = (0, 1), where = (segment != -1))
        
        rgb_means.append(rgb_mean) 

    rgb_means_lesion = np.mean(rgb_means[1:],axis=0)
    rgb_means_skin = np.mean(rgb_means[0])

    F1, F2, F3 = rgb_means_lesion/sum(rgb_means_lesion)
    F10, F11, F12 = rgb_means_lesion - rgb_means_skin
        
    return F1, F2, F3, F10, F11, F12
