"""
Data Clean
imapinvasives
Invasive species data
"""
import pandas as pd
import numpy as np

#Clean imapinvasives
file_name = "imapinvasives_bugs.csv"
df = pd.read_csv(file_name)
df = df[['x','y','common_name','number_found',\
                 'confirmed_ind','observation_date',\
                'species_type','growth_habit','county']]
df = df.rename(columns = {'x':'lon','y':'lat'})


df = df.loc[(df['county'] == "Queens") | (df['county'] == "New York") | \
            (df['county'] == "Kings") | (df['county'] == "Bronx") | \
            (df['county'] == "Richmond")]

df = df.loc[df['confirmed_ind']==True]
df = df.loc[df['growth_habit']=='Plant Pest']

df['number_found'] = df['number_found'].fillna(1)
df = df.drop(columns = ['confirmed_ind','growth_habit','species_type'])
#df['common_name'].unique()
df['Start'] = df['common_name'].apply(lambda s: s[0:3])
pdf = df[['Start']]
print(pdf.head(5))
#df.to_csv("invasive_nyc.csv",encoding='utf-8', index=False)
#df.info()

