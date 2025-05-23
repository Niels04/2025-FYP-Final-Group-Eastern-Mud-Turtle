from tqdm import tqdm
import pandas as pd
from pathlib import Path

#_________When importing from main_baseline.py the imports have to be changed like this____________
from util.img_util import ImageDataLoader as IDL
from util.img_util import rate_hair as rH
from util.feature_A import fA_extractor, fA_formula
from util.feature_B import fB_extractor, fB_formula
from util.feature_C import fC_extractor, fC_formula
from util.feature_D import fD_formula
from util.feature_BV import fBV_extractor
from util.feature_cheese import fCHEESE_extractor as fCH_extractor
from util.feature_snowflake import fSNOWFLAKE_extractor as fS_extractor

# from img_util import ImageDataLoader as IDL
# from hair_rating import rate_hair as rH
# from feature_A import fA_extractor
# from feature_B import fB_extractor
# from feature_C import fC_extractor
# from feature_BV import fBV_extractor
# from feature_cheese import fCHEESE_extractor as fCH_extractor
# from feature_snowflake import fSNOWFLAKE_extractor as fS_extractor

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" #obtain data directory

# set up relevant directories
img_dir = str(_DATA_DIR / "lesion_imgs/")
mask_dir = str(_DATA_DIR / "lesion_masks/")
metadata_dir = str(_DATA_DIR / "metadata.csv")
features_dir = str(_DATA_DIR / "features.csv")

def normalizeMinMax(column:pd.Series) -> pd.Series:
    #min and max from the column
    minVal = column.min()
    maxVal = column.max()

    #apply min-max scaling
    return ((column - minVal)/(maxVal - minVal))

# hepler function for main python script
def extract(img_dir, mask_dir= None, metadata_dir= None, features_dir= None, base_model= True, formula_features= False) -> pd.DataFrame:
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
    :param formula_features: Boolean value to indicate if the features needed to run the formula-based prediction will
                              be extracted, defaulted to False. If set to \"only\", the program will extract EXCLUSIVELY
                              features needed for the formula classifier to run, ignoring other parameters.

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
    for img_rgb, _, mask, mask_gs, name in tqdm(data_loader):

        # get patient_id and lesion_id from the filename
        name_split = name.split('_')
        patient_ID = '_'.join(name_split[:2])
        lesion_ID = int(name_split[2])

        # get full ID
        pat_les_ID = '_'.join(name_split[:3])

        if formula_features == 'only':

            A_val = fA_formula(mask)
            B_val = fB_formula(mask)
            C_val = fC_formula(img_rgb, mask)
            D_val = fD_formula()

            # create the new column for the final csv file
            datapoint = [name, patient_ID, lesion_ID, pat_les_ID, A_val, B_val, C_val, D_val]
        
        # extract the features with the proper functions
        else:
            fA_score, _ = fA_extractor(mask_gs)                # asymmetry - roundness of image
            fB_score = fB_extractor(mask)                      # border irregularity - compactness of image
            fC_score = fC_extractor(img_rgb, mask)             # color - amount of different colors in image
            hair_label = rH(img_rgb)                           # hair label (0 - 1 - 2 based on hair amount)

            if not base_model:
                fBV_score = fBV_extractor(img_rgb, mask)       # blue veil - amount of blue-ish pixels in lesion
                fCHEESE_score = fCH_extractor(mask)            # cheese - number of clusters in the mask of the lesion
                fSNOW_score = fS_extractor(img_rgb, mask)      # snowflake - checks if image hase white-ish pixels     

            if formula_features:
                A_val = fA_formula(mask)
                B_val = fB_formula(mask)
                C_val = fC_formula(img_rgb, mask)
                D_val = fD_formula()


            # create the new column for the final csv file

            datapoint = [name, patient_ID, lesion_ID, pat_les_ID, hair_label, fA_score, fB_score, fC_score]
            
            if not base_model:
                datapoint += [fBV_score, fCHEESE_score, fSNOW_score]
            
            if formula_features:
                datapoint += [A_val, B_val, C_val, D_val]
        
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

    if formula_features == 'only':

        col_names = ['img_id', 'patient_id', 'lesion_id', 'pat_les_ID', 'A_val', 'B_val', 'C_val', 'D_val']

        cd.columns = col_names

    else:

        col_names = ['img_id', 'patient_id', 'lesion_id', 'pat_les_ID', 'hair_label','fA_score', 'fB_score', 'fC_score']

        if not base_model:
            col_names += ['fBV_score', 'fCH_score', 'fS_score']
        
        if formula_features:
            col_names += ['A_val', 'B_val', 'C_val', 'D_val']

        if metadata_dir:
            col_names.append('true_melanoma_label')

        cd.columns = col_names

        # normalize the features that aren't between 0 and 1 already

        cd["fC_score"] = normalizeMinMax(cd["fC_score"])

        if not base_model:
            cd["fCH_score"] = normalizeMinMax(cd["fCH_score"])


    # save df if specified
    if features_dir:
        cd.to_csv(features_dir, index= False)

    return cd

if __name__ == "__main__":
    extract(img_dir, mask_dir, metadata_dir, features_dir, base_model= False, function_features= True)