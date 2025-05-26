#Lets make it a bit easier for us to look through the metadata csv
#The images folder should only contain the images of the lesions without subfolders.
import pandas as pd
md=pd.read_csv('data\metadata.csv',header=0)
md=md.sort_values(by=['patient_id'])
#MEL                            Point       BV
#img_id='PAT_680_1289_182.png'  #3.2   0.0003440416290371135
#img_id='PAT_495_941_26.png'    #3     0
#img_id='PAT_995_1867_165.png'  #4       0.31521756811712076
#Not MEL
#img_id='PAT_983_1853_985.png'  #2.4     0
#img_id='PAT_473_911_630.png'   #3.3     0
#img_id='PAT_809_1527_902.png'  #1.1     0.010660980810234541
#img_id='PAT_849_1612_143.png'  #4.8     0.2406148820114926

BV=0.2406148820114926
BV_threshold=0.05
diameter1 = float(md.loc[md['img_id'] == img_id, 'diameter_1'].values[0]) if pd.notna(md.loc[md['img_id'] == img_id, 'diameter_1'].values[0]) else 0
diameter2 = float(md.loc[md['img_id'] == img_id, 'diameter_2'].values[0]) if pd.notna(md.loc[md['img_id'] == img_id, 'diameter_2'].values[0]) else 0
age = int(md.loc[md['img_id'] == img_id, 'age'].values[0])
itch = md.loc[md['img_id'] == img_id, 'itch'].values[0]
grew = md.loc[md['img_id'] == img_id, 'grew'].values[0]
hurt = md.loc[md['img_id'] == img_id, 'hurt'].values[0]
changed = md.loc[md['img_id'] == img_id, 'changed'].values[0]
bleed = md.loc[md['img_id'] == img_id, 'bleed'].values[0]
elevation = md.loc[md['img_id'] == img_id, 'elevation'].values[0]
x=sum([1.2 if diameter1>6 or diameter2>6 else 0, 0.8 if BV>BV_threshold else 0, 0.8 if age>44 else 0, 0.1 if itch=='True' else 0, 
                1 if grew=='True' or changed=='True' else 0, 0.1 if hurt=='True' else 0, 0.8 if bleed=='True' else 0, 
                0.2 if elevation=='True' else 0, ])
print(x)