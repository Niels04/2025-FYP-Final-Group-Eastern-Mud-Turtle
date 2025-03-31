#Lets make it a bit easier for us to look through the metadata csv
#The images folder should only contain the images of the lesions without subfolders.
import pandas as pd
df=pd.read_csv('data\metadata.csv',header=0)
df=df.sort_values(by=['patient_id'])
print(df)
df.to_csv('data\sorted_metadata.csv',index=False)