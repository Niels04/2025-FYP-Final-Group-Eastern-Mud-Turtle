from img_util import ImageDataLoader as IDL
from inpaint_util import removeHair as rH
from tqdm import tqdm
from feature_A import fA_extractor
from feature_B import fB_extractor
from feature_C import fC_extractor
from feature_BV import fBV_extractor
from feature_cheese import fCHEESE_extractor
import pandas as pd


# set up relevant directories
img_dir = '../data/lesion_imgs/'
mask_dir = '../data/lesion_masks/'
metadata_dir = "../data/metadata.csv"
features_dir = "../data/features.csv"

# def training_extract(img_dir, mask_dir, metadata_dir, features_dir, base_model= True) -> None:
#     """Auxiliary python code that, given a set of lesion images with relative masks and metadata file,
#     outputs in a csv the relevant information about the lesion, such as patient id and lesion id, 
#     together with the extracted features, thanks to the imported functions.
#     A binary indicator about wether or not the lesion is a melanoma is also saved.
#     Intended to be used to load test data.

#     :param img_dir: The directory of the lesion images to be processed.
#     :param mask_dir: The directory of the relative mask images.
#     :param metadata_dir: The directory of the csv file containing information on the lesions.
#     :param features_dir: The directory in which the csv file containing extracted features will be saved
#     :param base_model: Boolean value to indicate if only the base features will be extracted, defaulted to True.

#     """
#     # read the relevant csv files
#     md = pd.read_csv(metadata_dir)
#     cd = pd.read_csv(features_dir)

#     # appropriately name columns
#     if base_model:
#         cd.columns= ['img_id', 'patient_id', 'lesion_id', 'pat_les_ID', 'fA_score', 'fB_score', 'fC_score', 'true_melanoma_label']
    
#     else:
#         cd.columns= ['img_id', 'patient_ID', 'lesion_ID', 'pat_les_ID', 'fA_score', 'fB_score', 'fC_score', 'fBV_score', 'fCHEESE_score', 'true_melanoma_label']

#     # print(len(md[md['diagnostic'] == 'MEL']) == len(cd[cd['true_melanoma_label'] == 1]))

#     # for idx, row in cd[cd['true_melanoma_label'] == 1].iterrows():
#     #     key = row['img_id']
        
#     #     # Find matching rows in df2
#     #     print(md[md['img_id'] == key]['diagnostic'].values[0])

#     # load up the images and relative masks
#     data_loader = IDL(img_dir, mask_dir)

#     # iterate through the pairs
#     for img_rgb, img_gray, mask, name in tqdm(data_loader):

#         # get patient_id and lesion_id from the filename - ALTERNATIVE: get them from metadata csv
#         name_split = name.split('_')
#         patient_ID = '_'.join(name_split[:2])
#         lesion_ID = int(name_split[2])

#         # get full ID
#         pat_les_ID = '_'.join(name_split[:3])
        
#         # remove the hair from the image
#         # _, _, img_rgb_nH = rH(img_rgb, img_gray)


#         # extract the features with the proper functions

#         fA_score = fA_extractor(mask)                      # asymmetry - roundness of image
#         fB_score = fB_extractor(mask)                      # border irregularity - compactness of image
#         fC_score = fC_extractor(img_rgb, mask)             # color - amount of different colors in image

#         if not base_model:
#             fBV_score = fBV_extractor(img_rgb, mask)       # blue veil - amount of blue-ish pixels in lesion
#             fCHEESE_score = fCHEESE_extractor(mask)        # cheese - number of clusters in the mask of the lesion


#         # get the actual diagnosis for the lesion, and convert it in a 
#         # binary value to indicate if it is melanoma or not
#         diagnosis = md[md['img_id'] == name]['diagnostic'].values[0]
#         true_label = 1 if diagnosis == 'MEL' else 0

#         # create the new column for the final csv file
#         if base_model:
#             datapoint = [name, patient_ID, lesion_ID, pat_les_ID, fA_score, fB_score, fC_score, true_label]
        
#         else:
#             datapoint = [name, patient_ID, lesion_ID, pat_les_ID, fA_score, fB_score, fC_score, fBV_score, fCHEESE_score, true_label]

#         cd.loc[len(cd)] = datapoint

#     # save the updated classified.csv
#     cd.to_csv(features_dir)

#     # TEMPORARY: how many name mismatches (48, out of 2000+)
#     print(f'Out of {len(data_loader)} image - mask pairs, {data_loader.lost} were lost due to name discrepancies.')


# hepler function for main python script
def extract(img_dir, mask_dir= None, metadata_dir= None, features_dir= None, base_model= True) -> pd.DataFrame:
    """Function to create, return and optionally save a Pandas data frame that stores information about 
    all image-maks pairs in the given directory.
    If only one directory is specified, the function assumes that both masks and images are found in it.
    By default, no metadata file will be loaded, the data frame will not be saved and only the base
    features will be extracted.
    To modify that, refer to the parameter documentation below.
    
    :param img_dir: The directory of the lesion images to be processed.
    :param mask_dir: The directory of the mask images, defaulted to None.
    :param metadata_dir: The directory of the metadata csv file containing the true melanoma label for the images,
                         with the file name under the column 'img_id', and the true label under the column 'diagnostic'.
                         Default is None, in which case no metadata file will be read and the resulting csv will not
                         contain the 'true_melanoma_label' column.
    :param features_dir: The directory in which to save the resulting data frame as a csv file.
                         Default is None, in which case the data frame will not be saved, but simply returned. 
    :param base_model: Boolean value to indicate if only the base features will be extracted, defaulted to True.
    
    :return: Pandas data frame with name of the file, patient and lesion id, value for each extracted feature, 
             for all images in the given directories.
             
    """

    # load up the images and relative masks
    if mask_dir:
        data_loader = IDL(img_dir, mask_dir)
    else:
        data_loader = IDL(img_dir, img_dir)


    # set up data frames according to parameters

    if metadata_dir:
        md = pd.read_csv(metadata_dir)
    
    cd = pd.DataFrame()


    # set up rows array
    rows = []

    # iterate through the pairs
    for img_rgb, img_gray, mask, name in data_loader:

        # get patient_id and lesion_id from the filename
        name_split = name.split('_')
        patient_ID = '_'.join(name_split[:2])
        lesion_ID = int(name_split[2])

        # get full ID
        pat_les_ID = '_'.join(name_split[:3])


        # extract the features with the proper functions

        fA_score = fA_extractor(mask)                      # asymmetry - roundness of image
        fB_score = fB_extractor(mask)                      # border irregularity - compactness of image
        fC_score = fC_extractor(img_rgb, mask)             # color - amount of different colors in image

        if not base_model:
            fBV_score = fBV_extractor(img_rgb, mask)       # blue veil - amount of blue-ish pixels in lesion
            fCHEESE_score = fCHEESE_extractor(mask)        # cheese - number of clusters in the mask of the lesion


        # create the new column for the final csv file

        datapoint = [name, patient_ID, lesion_ID, pat_les_ID, fA_score, fB_score, fC_score]
        
        if not base_model:
            datapoint += [fBV_score, fCHEESE_score]
        
        # get true melanoma label if metadata file is given
        if metadata_dir:

            # binary value to indicate if it is melanoma or not
            diagnosis = md[md['img_id'] == name]['diagnostic'].values[0]
            true_label = 1 if diagnosis == 'MEL' else 0

            # add label to datapoint
            datapoint.append(true_label)

        rows.append(datapoint)
    
    # add rows to data frame
    cd = pd.concat([cd, pd.DataFrame(rows)])


    # appropriately name columns

    col_names = ['img_id', 'patient_id', 'lesion_id', 'pat_les_ID', 'fA_score', 'fB_score', 'fC_score']

    if not base_model:
        col_names += ['fBV_score', 'fCHEESE_score']

    if metadata_dir:
        col_names.append('true_melanoma_label')

    cd.columns = col_names

    # save df if specified
    if features_dir:
        cd.to_csv(features_dir)
    
    return cd





