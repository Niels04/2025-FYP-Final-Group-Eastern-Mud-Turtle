import os
import random
import cv2

# ignore warnings - needed in cases of inconsistent image - mask names
cv2.setLogLevel(0)

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
    
    :return: img_rgb, img_gray, mask, name when iterated through.
    
    """
    def __init__(self, img_directory, mask_directory, transform=None):

        # set up the lists to store image and mask file names
        self.transform = transform
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
                mask = readImageFile(self.mask_list[i], is_mask= True)
            
            except Exception:
                self.lost += 1
                continue

            if self.transform:
                img_rgb = self.transform(img_rgb)
                img_gray = self.transform(img_gray)
            
            #obtain file name
            name = self.img_list[i].split("/")[-1]

            # yield necessary informations
            yield img_rgb, img_gray, mask, name