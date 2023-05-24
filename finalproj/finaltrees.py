"""
    Name: Min Joo Jeong
    Email: minjoo.jeong61@myhunter.cuny.edu
    Resources: hw11 answers
    uses nyc open data (see links)
    used jupyter notebooks for site
    SITE:
    https://minjoojeong.github.io/NYC-Tree-Health-Prediction/
"""


import pandas as pd
import re
import json
import numpy as np
from sodapy import Socrata


def tree2015():
"""
2015 trees using api call
huge dataset be warned
return df
2015 Street Tree Census
https://data.cityofnewyork.us/resource/uvpi-gqnh.json
"""
    # Unauthenticated client only works with public data sets. Note 'None'
    # in place of application token, and no username or password:
    client = Socrata("data.cityofnewyork.us", None)
    data = client.get("uvpi-gqnh", limit=700000)
    #data = client.get("uvpi-gqnh", limit=1200)
    key_dict = ['tree_id','latitude','longitude'\
                       ,'tree_dbh','status','health',\
                       'spc_latin','borough','nta_name']
    for item in data:
        for k,v in list(item.items()):
            if k not in key_dict:
                del item[k]
    #clean JSON            
    for item in data:
        for key, value in item.items():
            if key == 'spc_latin':
                new_v = value.split()
                genus = new_v[0]
                item[key] = genus
    json_obj = json.dumps(data)

    #Trees 2015 Health Data
    import pandas as pd
    import numpy as np
    treedf = pd.read_json(json_obj)
    tree2015 = treedf #the cleaned forestry json
    tree2015['health'] = tree2015['health'].fillna('Dead')
    tree2015['tpcondition'] = tree2015['health']
    tree2015['health'] = tree2015['health']
    dictionary = {'Excellent':3,'Good':3,'Fair':2,'Poor':1,'Dead':0}
    tree2015 = tree2015.replace({'health':dictionary})
    tree2015['spc_latin'] = tree2015['spc_latin'].fillna('Dead')
    tree2015 = tree2015.drop(columns = ['status'])
    return tree2015

def tree_2022():
    """
    Forestry Dataset
    https://data.cityofnewyork.us/resource/sivq-4tyd.json
    """
    #documentation for sodapy: 
    #https://dev.socrata.com/foundry/data.cityofnewyork.us/hn5i-inap 
    # Unauthenticated client only works with public data sets. Note 'None'
    # in place of application token, and no username or password:
    client = Socrata("data.cityofnewyork.us", None)
    #HUGE dataset unfortunately
    data = client.get("hn5i-inap", limit=1200000)
    #data = client.get("hn5i-inap", limit=1200)

    #cleaning JSON
    key_dict = ['objectid','dbh','tpcondition',\
                'geometry','genusspecies','createddate','updateddate']
    for item in data:
        for k,v in list(item.items()):
            if k not in key_dict:
                del item[k]

    for item in data:
        for key, value in item.items():
            if key == 'geometry':
                point = 'POINT ('
                new_v = value.replace(point,'')
                new_v = new_v.replace(')','').replace(' ',',')
                item['geometry'] = new_v
            if key == 'genusspecies':
                new_v = value.split()
                genus = new_v[0]
                item['genusspecies'] = genus
            if key == 'createddate' or key == 'updateddate':
                pattern = r'\..*$'
                co = re.compile(pattern)
                new_v = co.sub('',value)
                item[key] = new_v
    print(data[0]) #print json object
    json_obj = json.dumps(data)

    #Trees df cleaning
    import pandas as pd
    import numpy as np
    forestry = pd.read_json(json_obj)
    #health
    forestry['tpcondition'] = forestry['tpcondition'].fillna('Dead')
    dictionary = {'Excellent':3,'Good':3,'Fair':2,'Poor':1,\
                  'Dead':0, 'Critical':1,'Unknown':None}
    forestry['health_update'] = forestry['tpcondition']
    forestry = forestry.replace({'health_update':dictionary})
    #forestry['health'] = forestry['health'].astype(str).astype(int)
    #genus
    forestry['genusspecies'] = forestry['genusspecies'].fillna('Dead')
    #lat lon separate and convert to numbers
    #forestry = forestry['dbh'].astype(str).astype(int)
    forestry[['longitude','latitude']] = forestry.geometry.str.split(',',expand=True)
    forestry['longitude'] = forestry['longitude'].astype(float)
    forestry['latitude'] = forestry['latitude'].astype(float)
    forestry = forestry.drop(columns = ['tpcondition','geometry'])
    #date conversion
    forestry['createddate'] = pd.to_datetime(forestry['createddate'])
    forestry['updateddate'] = pd.to_datetime(forestry['updateddate'])

    #rename columns
    forestry = forestry.rename(columns = {'objectid':'tree_id','dbh':'tree_dbh',\
                                         'genusspecies':'spc_latin'})
    return forestry


tree2015 = tree2015() #call tree2015 function
tree2015.info()
tree2015.head()
#tree_2015 summary           

forestry = tree_2022() #make df
"""
visualize correlations for EDA
"""
import seaborn as sn
import matplotlib.pyplot as plt

corr_tree = tree_2015[['latitude','longitude','tree_dbh','health']]
corr_matrix = corr_tree.corr()
sn.heatmap(corr_matrix, annot=True)
plt.show()

def updated_trees(tree2015,forestry):
    """
    merge trees based on location
    """
    tree2015['location'] = list(zip(tree2015.latitude, tree2015.longitude))
    forestry['longitude'] = forestry['longitude'].round(decimals = 7)
    forestry['latitude'] = forestry['latitude'].round(decimals = 7)
    forestry['location'] = list(zip(forestry.latitude, forestry.longitude))
    sametrees = pd.merge(tree2015, forestry, on=['location'])
    sametrees = sametrees.loc[abs(sametrees['tree_dbh_x']-sametrees['tree_dbh_y'])<=2]
    sametrees = sametrees.loc[(sametrees['spc_latin_x']==sametrees['spc_latin_y']) | \
                          (sametrees['spc_latin_y']=='Unknown')]
    sametrees.loc[sametrees['spc_latin_y'] == 'Unknown', 'spc_latin_y'] = 'Dead'
    sametrees.loc[sametrees['spc_latin_y'] == 'Dead', 'health'] = 0
    return sametrees

sametrees = updated_trees(tree2015, forestry)
sametrees.info()
sametrees.head(5)

"""
visualize correlations for EDA
"""
import seaborn as sn
import matplotlib.pyplot as plt


corr_tree = sametrees

corr_matrix = corr_tree.corr()

sn.heatmap(corr_matrix, annot=True)
plt.show()
print(sametrees.spc_latin_x.unique()) #print unique genus

"""
clean the tree species data
"""
import pandas as pd

def clean_genus():
    """
    using USDA species search data
    https://plants.usda.gov/home/groupSearch
    awful website be warned, so csv files provided
    """
    gymno = pd.read_csv('gymnosperm.csv',skiprows=4)
    angio = pd.read_csv('dicot.csv',skiprows=4)
    #gymno
    gymno = gymno[['Scientific Name']]
    gymno['Angiosperm'] = 0 #not angiosperm
    gymno['genus'] = gymno['Scientific Name'].str.split(' ',1).str[0]
    gymno = gymno.drop(columns = ['Scientific Name'])
    gymno = gymno.drop_duplicates(subset=['genus'])
    #angio
    angio = angio[['Scientific Name']]
    angio['Angiosperm'] = 1
    angio['genus'] = angio['Scientific Name'].str.split(' ',1).str[0]
    angio = angio.drop(columns = ['Scientific Name'])
    angio = angio.drop_duplicates(subset=['genus'])
    #merge angiosperm data with gymnosperm
    tree_class = gymno.merge(angio, how='outer')
    return tree_class

tree_class = clean_genus()
#now has angiosperm data
#small datacleaning of classify tree
sametrees = sametrees.rename(columns={'spc_latin_x': 'genus'})
classifytree = pd.merge(sametrees, tree_class, on=['genus'])
classifytree = classifytree[['tree_id_x','tree_dbh_x','health','health_update','Angiosperm']]
classifytree = classifytree[['tree_id_x','tree_dbh_x','health','health_update','Angiosperm']]
classifytree = classifytree.rename(columns={'tree_id_x': 'tree_id','tree_dbh_x':'diameter'})

classifytree.head() #display cleaned data

"""
Use random forest classifier
"""
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def split_data(df, x_cols, y_cols, test_size=0.25, random_state=2023):
    """
    standard split test from hw
    """
    x_train, x_test, y_train, y_test = train_test_split(df[x_cols],
                                                        df[y_cols],
                                                        test_size=test_size,
                                                        random_state=random_state)
    return x_train, x_test, y_train, y_test

def fit_forest(x_train, y_train, xes,yes):
    """
    uses random forest
    display accuracy, confusion matrix
    """
    model = RandomForestClassifier(n_estimators=100, random_state=8)
    model.fit(x_train, y_train)
    mod_pkl = pickle.dumps(model)
    y_true = yes
    mod = pickle.loads(mod_pkl)
    y_pred = mod.predict(xes)
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_true, y_pred)
    return accuracy, confusion

xes = ['diameter','health','Angiosperm']
x_train,x_test,y_train,y_test = split_data(classifytree,xes,'health_update',0.25,223)
fit_forest(x_train, y_train, x_test, y_test)


"""
Mapping Aspect
"""
#import for mapping
import pandas as pd
import folium 
from folium.plugins import StripePattern
import geopandas as gpd
import numpy as np
#nyc trees
#merge angiosperm data with our csv
tree_class = gymno.merge(angio, how='outer')
sametrees = sametrees.rename(columns={'spc_latin_x': 'genus'})
sametrees = pd.merge(sametrees, tree_class, on=['genus'])
keep_col = ['tree_id_x','tree_dbh_x','health','genus','nta_name','latitude_x',\
           'tpcondition','location','updateddate','health_update','Angiosperm']
nyc_tree = sametrees[keep_col]
nta = 'NTA map.geojson'
nta_geo = gpd.read_file(nta)
# We read the file rename
nta_geo = nta_geo.rename(columns = {"ntaname":"nta_name"})
nyc_tree = nta_geo.merge(tree2015, on = 'nta_name')
#group by health
groupTree = nyc_tree.groupby('nta_name')['health'].mean().reset_index()
groupTree.head()
groupTree_geo = nta_geo.merge(groupTree, on = 'nta_name')
groupTree_geo.head()

nycMap = folium.Map(location=[40.723092, -73.844215],\
                    tiles='cartodbpositron',\
                    zoom_start=10)
#set up choropleth
folium.Choropleth(geo_data=groupTree_geo,
                  data=groupTree_geo,
                  columns=['nta_name','health'],
                  key_on='feature.properties.nta_name',
                  fill_color='BuPu',
                  fill_opacity=1,
                  line_opacity=0.2,
                  legend_name="Health",).add_to(nycMap)
nycMap #display map
