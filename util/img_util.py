import os
import numpy as np
import cv2

import warnings
warnings.catch_warnings 

# ignore warnings - needed in cases of inconsistent image - mask names
cv2.setLogLevel(0)

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
    
    # get image shape and convert to grayscale
    image_size = image.shape[:2]
    img = image.mean(-1)

    # optionally blur the image (heavily suggested)
    if blur:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # --------------------------------------------------------

    # enhance dark structures
    kernel = np.ones((3,3),np.uint8)
    img_filt = cv2.morphologyEx(np.uint8(img), cv2.MORPH_BLACKHAT, kernel) 
    img_filt = np.where(img_filt > 15, img_filt, 0)
    
    # dilate found structures
    kernel = np.ones((4,4),np.uint8)
    img_filt = cv2.morphologyEx(img_filt, cv2.MORPH_DILATE, kernel)
        
    # -------------------------------------------------------- 
    
    # mask dark regions only, then dilate them
    dark_spots = (img < dst).astype(np.uint8)
    kernel = np.ones((4,4),np.uint8)
    dark_spots = cv2.morphologyEx(dark_spots, cv2.MORPH_DILATE, kernel)
    
    # keep only intersections
    img_filt = img_filt * dark_spots
    
    # -------------------------------------------------------- 
    
    # detect lines starting from edges and dark spots
    lines = cv2.HoughLinesP(img_filt, cv2.HOUGH_PROBABILISTIC, np.pi / 90, 20, None, 1, 20)

    # iterate through lines, generate pixel coordinates for each
    if lines is not None:
        lines = lines.reshape(-1, 4)
        N_lines = lines.shape[0]

        # exclude short lines
        lines_to_interp = []
        for ind in range(N_lines):
            line = lines[ind, :]
            x, y = fill_line(line[0::2], line[1::2], 1)
            lines_to_interp.append( (x, y) )

    # if there are no lines, create black mask      
    else:
        lines_to_interp = []
        img_filt = np.zeros(image_size)

                
    # -------------------------------------------------------- 
    # include line coordinates in the mask
    Mask = np.zeros_like(img_filt)
    for (x, y) in lines_to_interp:
        Mask[y, x] = 1

    # dilate final mask
    kernel = np.ones((3,3),np.uint8)
    Mask = cv2.morphologyEx(Mask, cv2.MORPH_DILATE, kernel)
    Mask = Mask.astype(float)  

    # --------------------------------------------------------
    
    # if no non-zero pixels are found, the mask is reset
    i, j = np.where( Mask != 0 )

    if i.size == 0:
        Mask = np.zeros(image_size)

    # compute hair ratio and appropriately label image
    ratio = np.count_nonzero(Mask) / (image_size[0] * image_size[1])

    if ratio >= t2:
        label = 2
    elif ratio >= t1:
        label = 1
    else:
        label = 0
              
    return ratio, label, Mask

def fill_line(x, y, step=1):
    
    # given two endpoints x = [x0, x1], y = [y0, y1], returns two lists containing
    # all intermediate pixel positions
    points = []

    # case for vertical line: x cords do not change:
    # add all y cords from min to max, use the same
    # x val for every point.
    if x[0] == x[1]:
        ys = np.arange(y.min(), y.max(), step)
        xs = np.repeat(x[0], ys.size)

    else:
        
        # compute slope of the line
        m = (y[1] - y[0]) / (x[1] - x[0])
        
        # get all x cords
        xs = np.arange(x[0], x[1], step * np.sign(x[1]-x[0]))

        # use y = y0 + m * (x - x0) to find y values
        ys = y[0] + m * (xs-x[0])
    
    return xs.astype(int), ys.astype(int)

def readImageFile(file_path, is_mask = False): # added is_mask parameter
    """Given a path, the corresponding image will be loaded in both rgb and grayscale color format.
    If the given image is indicated to be a mask, it will only be loaded in grayscale
    
    :param file_path: The image path
    :param is_mask: Defaulted to False, if set to True it only loads the mask image in grayscale
    
    :return: RGB and Grayscale image, or Grayscale mask.
    
    """
    # if we want to load a binary mask, we read the file in grayscale format
    if is_mask:
        
        mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        return mask
    
    # else we proceed normally

    # read image as an 8-bit array
    img_bgr = cv2.imread(file_path)

    # convert to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # convert the original image to grayscale
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    return img_rgb, img_gray


def saveImageFile(img_rgb, file_path):
    try:
        # convert BGR
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # save the image
        success = cv2.imwrite(file_path, img_bgr)
        if not success:
            print(f"Failed to save the image to {file_path}")
        return success

    except Exception as e:
        print(f"Error saving the image: {e}")
        return False


class ImageDataLoader:
    """Given a directory for images and a directory for masks, it stores the path names,
    And returns information about the image, along with the image itself, when iterated through.
    
    :param img_directory: The file path to the folder containing the images
    :param mask_directory: The file path to the folder containing the images
    
    :yield:s img_rgb, img_gray, mask, name when iterated through.
    
    """
    def __init__(self, img_directory, mask_directory):

        # set up the lists to store image and mask file names
        self.img_list = []
        self.mask_list = []

        # set up counter to keep track of the amount of name inconsistencies
        self.lost = 0
        
        # iterate through the masks folder to save the individual file paths
        for f in os.listdir(mask_directory):

            # if the file is in a valid format
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                
                # different directories or mask file
                if mask_directory != img_directory or '_mask' in f:

                    # get the full path of the mask
                    mask_path = os.path.join(mask_directory, f)

                    # get the name of the corresponding image
                    img_name = f.replace('_mask', '')

                    # get the full path of the image
                    img_path = os.path.join(img_directory, img_name)
                
                # same directories and image file
                else:

                    # get the full path of the image
                    img_path = os.path.join(img_directory, f)

                    # get the name of the corresponding mask
                    name, ext = os.path.splitext(f)
                    mask_name = f"{name}_mask{ext}"

                    # get the full path of the mask
                    mask_path = os.path.join(mask_directory, mask_name)

                # append mask and image to the relative lists
                self.img_list.append(img_path)
                self.mask_list.append(mask_path) 

        if not self.img_list:
            raise ValueError("No image files found in the directory.")

        # get the total number of files
        self.num_sample = len(self.img_list)
        
    def __len__(self):
        return self.num_sample

    def __iter__(self):
        for i in range(self.num_sample):
            
            # try to load the image and mask, if one of the two leads to no such file, update loss counter
            try:
                img_rgb, img_gray = readImageFile(self.img_list[i])
                mask_gs = readImageFile(self.mask_list[i], is_mask= True)
                mask = (mask_gs > 127).astype(np.uint8) # mask as binary
                # if the mask only contains 0s, update counter and skip
                unique_vals = np.unique(mask)
                if len(unique_vals) == 1 and int(unique_vals[0]) == 0:
                    self.lost += 1
                    continue
            
                #obtain file name
                name = self.img_list[i].split("\\")[-1]

                # yield necessary informations
                yield img_rgb, img_gray, mask, mask_gs, name 
            
            except Exception:
                self.lost += 1
                continue