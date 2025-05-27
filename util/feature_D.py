# temporary
import pandas as pd
def fD_formula(img_id, md, BV, BV_threshold = 0.05):
    """
    Extracts feature D which is used for the adjusted Stolz formula.
    Due to the fact the pictures are not dermoscope pictures we had to add
    such a parameters that are helpful in the prediction of Melanoma.

    :param img_id: image id to get the correct values from the metadata
    :param md: the pandas dataframe stored in metadata.csv
    :param BV: BlueVeils
    :param BV_threshold: ]0,1[, it determinetes the decision boundary for the BlueVeil

    :return: int ]0,5[ 
    """

    diameter1 = float(md.loc[md['img_id'] == img_id, 'diameter_1'].values[0]) if pd.notna(md.loc[md['img_id'] == img_id, 'diameter_1'].values) else 0
    diameter2 = float(md.loc[md['img_id'] == img_id, 'diameter_2'].values[0]) if pd.notna(md.loc[md['img_id'] == img_id, 'diameter_2'].values) else 0
    age = int(md.loc[md['img_id'] == img_id, 'age'].values[0])
    itch = md.loc[md['img_id'] == img_id, 'itch'].values[0]
    grew = md.loc[md['img_id'] == img_id, 'grew'].values[0]
    hurt = md.loc[md['img_id'] == img_id, 'hurt'].values[0]
    changed = md.loc[md['img_id'] == img_id, 'changed'].values[0]
    bleed = md.loc[md['img_id'] == img_id, 'bleed'].values[0]
    elevation = md.loc[md['img_id'] == img_id, 'elevation'].values[0]

    return sum([1.2 if diameter1>6 or diameter2>6 else 0, 0.8 if BV>BV_threshold else 0, 0.8 if age>44 else 0, 0.1 if itch=='True' else 0, 
                1 if grew=='True' or changed=='True' else 0, 0.1 if hurt=='True' else 0, 0.8 if bleed=='True' else 0, 
                0.2 if elevation=='False' else 0, ])

                            #Points

###age: 45 is the peak          0.8
###Size 6, 6                    1.2
###itch 1-2                     0.1
# #grew 5                       
# #hurt 1-2                     0.1
# #changed 5                    
# #bleed 5                      0.8
# #elevation ? but rather flat  0.2

#we merge grew and change       1

