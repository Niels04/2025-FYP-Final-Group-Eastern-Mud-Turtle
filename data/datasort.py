#Lets make it a bit easier for us to look through the metadata csv

import pandas as pd
df=pd.read_csv('data\metadata.csv',header=0)
df=df.sort_values(by=['patient_id'])
print(df)
df.to_csv('data\sorted_metadata.csv',index=False)