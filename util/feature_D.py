# temporary
import pandas as pd
def fD_formula(img_id,metadata_dir,BV, CH, BV_threshold = 0.05, CH_threshold = 0.05):
    """
    Extracts feature D which is used for the Stolz formula.
    Due to the fact the pictures are not dermoscope pictures we had to add
    such a parameters that are helpful in the prediction of Melanoma.

    :param img_id: image id to get the correct values from the metadata
    :param metadata_dir: the directory of the metadata.csv
    :param BV: BlueVails (used in the original formula)
    :param CH: The CheeseFeature (used in the original formula)
    :param BV_threshold: ]0,1[, it determinetes the deceision boundary for the BlueVail
    :param CH_threshold: ]0,1[, it determinetes the deceision boundary for the CheeseFeature
    
    :return: int ]1,5[ 
    """
    md = pd.read_csv(metadata_dir)
    diameter1 = float(md.loc[md['img_id'] == img_id, 'diameter_1'].values[0]) if pd.notna(md.loc[md['img_id'] == img_id, 'diameter_1'].values[0]) else 0
    diameter2 = float(md.loc[md['img_id'] == img_id, 'diameter_2'].values[0]) if pd.notna(md.loc[md['img_id'] == img_id, 'diameter_2'].values[0]) else 0
    age = md.loc[md['img_id'] == img_id, 'age'].values[0]
    itch = md.loc[md['img_id'] == img_id, 'itch'].values[0]
    grew = md.loc[md['img_id'] == img_id, 'grew'].values[0]
    hurt = md.loc[md['img_id'] == img_id, 'hurt'].values[0]
    changed = md.loc[md['img_id'] == img_id, 'changed'].values[0]
    bleed = md.loc[md['img_id'] == img_id, 'bleed'].values[0]
    elevation = md.loc[md['img_id'] == img_id, 'elevation'].values[0]
    return sum([1 if diameter1>6 or diameter2>6 else 0, 1 if BV>BV_threshold else 0, 1 if CH>CH_threshold else 0])

###age: 45 is the peak
###Size 6, 6
###itch 1-2
# #grew 5
# #hurt 1-2
# #changed 5
# #bleed 5
# #elevation ? but rather flat
### region: by the book: tight, chest