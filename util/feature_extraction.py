from img_util import ImageDataLoader as IDL
from inpaint_util import removeHair as rH
from tqdm import tqdm
from feature_A import fA_extractor
from feature_B import fB_extractor
from feature_C import fC_extractor
import pandas as pd


"""Auxiliary python file that, given a set of lesion images with relative masks and metadata,
outputs in a csv the relevant information about the lesion, such as patient id and lesion id, 
with the computed features, thanks to the imported functions.
A binary indicator about wether or not the lesion is a melanoma is also saved."""

# set up relevant directories
img_dir = '../data/lesion_imgs/'
mask_dir = '../data/lesion_masks/'
metadata_dir = "../data/metadata.csv"
features_dir = "../data/features.csv"

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




