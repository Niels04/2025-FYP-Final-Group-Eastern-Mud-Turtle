from img_util import ImageDataLoader as IDL
from inpaint_util import removeHair as rH
from tqdm import tqdm
from feature_A import fA_extractor
from feature_B import fB_extractor
from feature_C import fC_extractor
import pandas as pd


"""Auxiliary python code that, given a set of lesion images with relative masks and metadata,
outputs in a csv the relevant information about the lesion, such as patient id and lesion id, 
with the computed features, thanks to the imported functions.
A binary indicator about wether or not the lesion is a melanoma is also saved."""

# set up relevant directories
img_dir = '../data/lesion_imgs/'
mask_dir = '../data/lesion_masks/'
metadata_dir = "../data/metadata.csv"
features_dir = "../data/features.csv"

def training_extract(img_dir, mask_dir, metadata_dir, features_dir) -> None:
    # read the relevant csv files
    md = pd.read_csv(metadata_dir)
    cd = pd.read_csv(features_dir)

    # print(len(md[md['diagnostic'] == 'MEL']) == len(cd[cd['true_melanoma_label'] == 1]))

    # for idx, row in cd[cd['true_melanoma_label'] == 1].iterrows():
    #     key = row['img_id']
        
    #     # Find matching rows in df2
    #     print(md[md['img_id'] == key]['diagnostic'].values[0])

    # load up the images and relative masks
    data_loader = IDL(img_dir, mask_dir)

    # iterate through the pairs
    for img_rgb, img_gray, mask, name in tqdm(data_loader):

        # get patient_id and lesion_id from the filename - ALTERNATIVE: get them from metadata csv
        name_split = name.split('_')
        patient_ID = '_'.join(name_split[:2])
        lesion_ID = int(name_split[2])

        # get full ID
        pat_les_ID = '_'.join(name_split[:3])
        
        # remove the hair from the image
        _, _, img_rgb_nH = rH(img_rgb, img_gray)

        # extract the features with the proper function
        fA_score = fA_extractor(mask)           # asymmetry - roundness of image
        fB_score = fB_extractor(mask)           # border irregularity - compactness of image
        fC_score = fC_extractor(img_rgb, mask)  # color - amount of different colors in image

        # get the actual diagnosis for the lesion, and convert it in a 
        # binary value to indicate if it is melanoma or not
        diagnosis = md[md['img_id'] == name]['diagnostic'].values[0]
        true_label = 1 if diagnosis == 'MEL' else 0

        # create the new column for the final csv file
        datapoint = [name, patient_ID, lesion_ID, pat_les_ID, fA_score, fB_score, fC_score, true_label]
        cd.loc[len(cd)] = datapoint

        # cd.to_csv(features_dir, index=False)

    # save the updated classified.csv
    cd.to_csv(features_dir, index=False)

    # TEMPORARY: how many name mismatches (48, out of 2000+)
    print(f'Out of {len(data_loader)} image - mask pairs, {data_loader.lost} were lost due to name discrepancies.')


# hepler function for main python script
def extract(img_dir, mask_dir= None) -> pd.DataFrame:
    """Function to create and return a Pandas data frame that stores information about all image - maks pairs in the given directory.
    If only one directory is specified, the function assumes that both masks and images are found in it.
    If two directories are given, the first one will be treated as the image directory, and the second one as the mask directory.
    The data frame contains, for each image, name of the file, patient and lesion id, value for all extracted features.
    
    :param img_dir: The directory of the lesion images to be processed.
    :param mask_dir: The directory of the mask images, defaulted to None.
    
    :return: Pandas data frame with name of the file, patient and lesion id, value for each extracted feature, 
             for all images in the given directories.
             
    """

    # load up the images and relative masks
    if mask_dir:
        data_loader = IDL(img_dir, mask_dir)
    else:
        data_loader = IDL(img_dir, img_dir)

    
    # set up data frame to store results
    cd = pd.DataFrame()

    # set up rows array
    rows = []

    # iterate through the pairs
    for img_rgb, img_gray, mask, name in data_loader:

        # get patient_id and lesion_id from the filename
        name_split = name.split('_')
        patient_ID = '_'.join(name_split[:2])
        lesion_ID = int(name_split[2])
        
        # remove the hair from the image
        _, _, img_rgb_nH = rH(img_rgb, img_gray)

        # extract the features with the proper function
        fA_score = fA_extractor(mask)           # asymmetry - roundness of image
        fB_score = fB_extractor(mask)           # border irregularity - compactness of image
        fC_score = fC_extractor(img_rgb, mask)  # color - amount of different colors in image

        # create the new column for the final csv file
        datapoint = [name, patient_ID, lesion_ID, fA_score, fB_score, fC_score]
        rows.append(datapoint)
    
    # add rows to data frame
    cd = pd.concat([cd, pd.DataFrame(rows)])

    # appropriately name columns
    cd.columns= ['name', 'patient_ID', 'lesion_ID', 'fA_score', 'fB_score', 'fC_score']

    return cd





