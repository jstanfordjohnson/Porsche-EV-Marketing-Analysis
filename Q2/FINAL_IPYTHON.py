#!/usr/bin/env python
# coding: utf-8

# In[169]:


import pandas as pd
import seaborn as sns
import zipcodes
import numpy as np


# In[170]:


charge_point = pd.read_csv("State_wise_data/alt_fuel_stations (Dec 13 2019)-2.csv")


# In[171]:


charge_point = charge_point[charge_point["EV Network"] != 'Tesla']
charge_point = charge_point[charge_point["EV Network"] != 'Tesla Destination']


# In[172]:


california_zip = [x for x in range(90000,100000)]


# In[173]:


california_original_zip = []
flag = 0
for x in california_zip:
    k = zipcodes.matching(str(x))
    flag = flag + 1
    print(flag)
    if len(k) > 0:
        if k[0]['state'] == 'CA':
            california_original_zip.append(x)


# In[174]:


california_cp_1 = charge_point.loc[charge_point["ZIP"].isin(california_original_zip)]


# In[175]:


aggregations = {
    "EV Level2 EVSE Num": "sum",
    "EV DC Fast Count": "sum"
}

california_ev_zip = california_cp_1.groupby("ZIP").aggregate(aggregations)
california_ev_zip


# In[176]:


#california_augment[california_augment["Fuel"] == "Battery Electric"]["Make"].value_counts()


# In[177]:


#california_augment


# In[178]:


california_augment = pd.read_csv("State_wise_data/california_VehicleCount_100118.csv")
california_augment


# In[179]:


california_augment = california_augment.loc[california_augment['Duty'] != 'Heavy']


california_augment


# In[180]:


with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(california_augment['Make'].value_counts())


# In[181]:


car_brands = california_augment["Make"].value_counts().index.tolist()


# In[182]:


car_brands


# In[183]:


d =  ['Acura',
 'Alfa Romeo',
 'Audi',
 'BMW',
 'Buick',
 'Cadillac',
 'Chevrolet',
 'Chrysler',
 'Dodge',
 'FIAT',
 'Ford',
 'Genesis',
 'GMC',
 'Honda',
 'Hyundai',
 'INFINITI',
 'Jaguar',
 'Jeep',
 'Kia',
 'Land Rover',
 'Lexus',
 'Lincoln',
 'Lotus',
 'Maserati',
 'Mazda',
 'Mercedes-Benz',
 'MINI',
 'Mitsubishi',
 'Nissan',
 'Porsche',
 'Ram',
 'smart',
 'Subaru',
 'Toyota',
 'Volkswagen',
 'Volvo',
'rolls-royce',
 'bentley',
 'ferrari',
 'tesla',
 'aston martin']


# In[184]:


len(d)
len(car_brands)


# In[185]:


for i in range(len(car_brands)):
    car_brands[i] = car_brands[i].lower()

for i in range(len(d)):
    d[i] = d[i].lower()
    


# In[186]:


def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 

matching = intersection(car_brands, d)


# In[187]:


non_matching = [x for x in car_brands if x not in matching]
#non_matching
#matching
#drop it


# In[188]:


dict4 = {'s': 32988.15789473684,
 'Acura': 60012.5,
 'Alfa Romeo': 30285.0,
 'Audi': 64005.6,
 'BMW': 72017.22222222222,
 'Buick': 32965.71428571428,
 'Cadillac': 59462.72727272727,
 'Chevrolet': 31975.68181818182,
 'Chrysler': 16030.833333333334,
 'Dodge': 14832.0,
 'FIAT': 12045.625,
 'Ford': 31512.105263157893,
 'Genesis': 49928.333333333336,
 'GMC': 34755.0,
 'Honda': 26968.18181818182,
 'Hyundai': 23524.090909090908,
 'INFINITI': 45746.25,
 'Jaguar': 55715.0,
 'Jeep': 15952.272727272728,
 'Kia': 28795.833333333332,
 'Land Rover': 59205.0,
 'Lexus': 54833.181818181816,
 'Lincoln': 46930.625,
 'Lotus': 96785.0,
 'Maserati': 104000.0,
 'Mazda': 25227.14285714286,
 'Mercedes-Benz': 56850.5,
 'MINI': 28625.0,
 'Mitsubishi': 21702.5,
 'Nissan': 33356.05263157895,
 'Porsche': 73450.0,
 'Ram': 16526.25,
 'smart': 24650.0,
 'Subaru': 26337.5,
 'Toyota': 32988.15789473684,
 'Volkswagen': 28467.272727272728,
 'Volvo': 45033.88888888889,
 'rolls-royce': 353290.0,
 'bentley': 235060.0,
 'ferrari': 285573.5,
 'tesla': 67860.0,
 'aston martin': 241174.0,
'saturn':30000,
'pontiac' :30000,
'mercury' : 30000,
'oldsmobile' : 30000,
'isuzu' : 25000,
'suzuki' : 20000,
'plymouth' : 20000,
'hummer' : 50000,
'geo' : 5000,
'saab' : 5000,
"gem" : 5000,
'clubcar' : 10000, 
'columbia' : 10000, 
'international' : 15000, 
'freightliner_trucks' : 30000, 
'workhorse' : 50000 ,
'mg' : 20000,
'tomberlin' : 5000,
'westerngolfcart' : 5000,
'dymacvehiclegroup' : 5000,
'vpg' : 10000,
'americancustomgolfcarts' : 5000, 
'ezgo' : 5000,
'amgeneral' : 300000
}


# In[189]:


dict4 = {key.lower() if isinstance(key, str) else key: value for key, value in dict4.items()}
dict4


# In[190]:


drop = ['clubcar','columbia','international','freightliner_trucks','workhorse','tomberlin','western golf cart','dymac vehicle group','american custom golfcarts','ezgo']


# In[191]:


california_augment['Make'] = california_augment['Make'].apply(lambda x: x.lower())


# In[192]:


california_augment[~california_augment['Make'].isin(drop)]


# In[193]:


with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(california_augment['Make'].value_counts())


# In[194]:


california_augment.drop(['AVERAGE_PRICE_OF_CARS','Unnamed: 8'], axis = 1, inplace = True)


# In[195]:


import numpy as np


# In[196]:


california_augment['AVG'] = california_augment['Make'].apply(lambda x: dict4.get(x,np.nan))
#california_augment['Fuel'].value_counts()


# In[197]:


california_augment = california_augment[(california_augment["ZIP"] != 'Other')]
california_augment = california_augment[(california_augment["ZIP"] != 'OOS')]


# In[198]:


california_augment["ZIP"] = california_augment["ZIP"].astype("int")


# In[199]:


california_augment["AVG"] = california_augment.groupby(["ZIP","Fuel"])['AVG'].transform(lambda x: x.fillna(x.mean()))


# In[200]:


california_augment.groupby("ZIP").aggregate({"Vehicles":"sum"})


# In[201]:


california_augment["AVG"] = california_augment.groupby(["ZIP"])['AVG'].transform(lambda x: x.fillna(x.mean()))


# In[202]:


california_augment.isnull().sum()


# In[203]:


california_augment["AVG"].fillna(california_augment["AVG"].mean(),inplace = True)


# In[204]:


def prefix(df,level):
    dict1 = dict()
    
    column = df.columns.tolist()
    
    column = column[1:]

    
    for i in column:
        dict1[i] = i.replace(" ","").lower() + level
    
    df = df.rename(columns=dict1)
    
    
    return df


# In[205]:


level1 = california_augment[california_augment["AVG"] <= 20000]


# In[206]:


level1 = level1.groupby(["ZIP","Fuel"]).aggregate({"Vehicles": "sum"}).unstack(level = -1)
level1 = level1['Vehicles']
level1 = level1.reset_index()


# In[207]:


level1 = prefix(level1,"0_20000")
level1.head()


# In[208]:


level2 = california_augment.loc[(california_augment["AVG"] > 20000) & (california_augment["AVG"] <= 30000)]


# In[209]:


level2 = level2.groupby(["ZIP","Fuel"]).aggregate({"Vehicles": "sum"}).unstack(level = -1)
level2 = level2['Vehicles']
level2 = level2.reset_index()


# In[210]:


level2 = prefix(level2,"20000_30000")
level2.head()


# In[211]:


level3 = california_augment.loc[(california_augment["AVG"] > 30000) & (california_augment["AVG"] <= 40000)]


# In[212]:


level3 = level3.groupby(["ZIP","Fuel"]).aggregate({"Vehicles": "sum"}).unstack(level = -1)
level3 = level3['Vehicles']
level3 = level3.reset_index()


# In[213]:


level3 = prefix(level3,"30000_40000")
level3.head()


# In[214]:


level4 = california_augment.loc[(california_augment["AVG"] > 40000) & (california_augment["AVG"] <= 70000)]
level4.head()


# In[215]:


level4 = level4.groupby(["ZIP","Fuel"]).aggregate({"Vehicles": "sum"}).unstack(level = -1)
level4 = level4['Vehicles']
level4 = level4.reset_index()


# In[216]:


level4 = prefix(level4,"40000_70000")
level4.head()


# In[217]:


level5 = california_augment.loc[(california_augment["AVG"] > 70000) & (california_augment["AVG"] <= 100000)]
level5.head()


# In[218]:


level5 = level5.groupby(["ZIP","Fuel"]).aggregate({"Vehicles": "sum"}).unstack(level = -1)
level5 = level5['Vehicles']
level5 = level5.reset_index()


# In[219]:


level5 = prefix(level5,"70000_100000")
level5.head()


# In[220]:


level6 = california_augment.loc[(california_augment["AVG"] > 100000)]
level6.head()


# In[221]:


level6 = level6.groupby(["ZIP","Fuel"]).aggregate({"Vehicles": "sum"}).unstack(level = -1)
level6 = level6['Vehicles']
level6 = level6.reset_index()


# In[222]:


level6 = prefix(level6,"100000")
level6.head()


# In[223]:


level1.fillna(0,inplace = True)
level2.fillna(0,inplace = True)
level3.fillna(0,inplace = True)
level4.fillna(0,inplace = True)
level5.fillna(0,inplace = True)
level6.fillna(0,inplace = True)


# In[224]:


level1 = level1.merge(level2, on = "ZIP", how = "outer")


# In[225]:


level1 = level1.merge(level3, on = "ZIP", how = "outer")


# In[226]:


level1 = level1.merge(level4, on = "ZIP", how = "outer")


# In[227]:


level1 = level1.merge(level5, on = "ZIP", how = "outer")


# In[228]:


level1 = level1.merge(level6, on = "ZIP", how = "outer")


# In[229]:


level1


# In[230]:


level1.fillna(0,inplace = True)


# In[231]:


level1["ZIP"] = level1["ZIP"].astype("int")


# In[232]:


california_augment['Vehicles']


# In[233]:


cd = california_augment.groupby(["ZIP","Fuel"]).aggregate({"Vehicles": "sum"})
cd


# In[234]:


cd


# In[235]:


cd.unstack(level = -1)


# In[236]:


cd = cd.unstack(level = -1)


# In[237]:


cd.fillna(0, inplace = True)


# In[238]:


cd = cd["Vehicles"]


# In[239]:


cd1 = cd


# In[240]:


cd1 = cd1.reset_index()


# In[241]:


cd1.sum()


# In[242]:


cd2 = cd1[cd1["ZIP"].isin(california_original_zip)]


# In[311]:


cd1


# In[244]:


cd2.head()


# In[245]:


california_ev_zip.head()


# In[246]:


cali_age_gender = pd.read_csv("State_wise_data//california/cali_age_gender_population_estimate/ACS_17_5YR_DP05_with_ann.csv")


# In[247]:


#cali_age_gender["GEO.id2"][1:1776].intersection(cd2["ZIP"])
#set(cali_age_gender["GEO.id2"][1:1776]).intersection(set(cd2["ZIP"]))
#series(list(set(cali_age_gender["GEO.id2"][1:1776]) & set(cd2["ZIP"])))
cali_age_gender["GEO.id2"][1:1776].astype("int").isin(cd2["ZIP"].astype("int")).value_counts()
cali_age_gender["GEO.id2"][1:1776].astype("int").isin(cd2["ZIP"].astype("int")).value_counts()


# In[248]:


cali_age_gender = cali_age_gender.iloc[1:1776]


# In[249]:


cali_age_gender[["GEO.id2","HC01_VC37","HC01_VC36"]]


# In[250]:


def new_select(df,parameters,new_parameters):
    df = df[parameters]
    df.columns = new_parameters
    df = df.astype("int")
    return df

def new_merge(df1,df2):
    df = df1.merge(df2, on = "ZIP", how = "inner")
    return df


    


# In[251]:


cali_age_gender2 = new_select(cali_age_gender,["GEO.id2","HC01_VC37","HC01_VC36"],["ZIP","FEMALE_ABOVE_18","MALE_ABOVE_18"])
cd3 = new_merge(cd2,cali_age_gender2)
cd3


# In[252]:


cali_education_age = pd.read_csv("State_wise_data//california/cali_total_education_by_different_Standards/ACS_17_5YR_S1501_with_ann.csv")


# In[253]:


cali_education_age = cali_education_age.iloc[1:]


# In[254]:


cali_education_age.head()


# In[255]:


cali_education_age["GEO.id2"] = cali_education_age["GEO.id2"].astype("int") 


# In[256]:


param = ["GEO.id2","HC01_EST_VC03","HC01_EST_VC04","HC01_EST_VC05","HC01_EST_VC06","HC01_EST_VC09","HC01_EST_VC10","HC01_EST_VC11","HC01_EST_VC12","HC01_EST_VC13","HC01_EST_VC14","HC01_EST_VC15"]
new_param = ["ZIP","LESS_HIGHSCHOOL_18_24","HIGHSCHOOL_18_24","SOMECOLLEGE_18_24","BACHELOR_18_24","HC01_EST_VC09","HC01_EST_VC10","HIGHSCHOOL_25_","SOMECOLLEGE_25_","ASSOCIATE_25_","BACHELOR_25_","MASTER_25_"]
cali_education_age2 = new_select(cali_education_age,param,new_param)
cali_education_age2.columns = new_param


# In[257]:


cali_education_age2.dtypes


# In[258]:


cali_education_age2['HIGHSCHOOL_25_'] = cali_education_age2['HC01_EST_VC09'] + cali_education_age2['HC01_EST_VC10']


# In[259]:


cali_education_age2.drop(['HC01_EST_VC09','HC01_EST_VC10'], axis = 1, inplace = True)


# In[260]:


cali_education_age2.head()


# In[261]:


cd4 = new_merge(cd3,cali_education_age2)


# In[262]:


cd4.head()


# In[263]:


cali_employ_income = pd.read_csv("State_wise_data/california/cali_income_and_employment/ACS_17_5YR_DP03_with_ann.csv")


# In[264]:


cali_employ_income = cali_employ_income.iloc[1:]


# In[265]:


cali_employ_income["GEO.id2"] = cali_employ_income["GEO.id2"].astype("int")


# In[266]:


param = ["GEO.id2","HC01_VC41","HC01_VC42","HC01_VC43","HC01_VC44","HC01_VC45","HC01_VC75","HC01_VC76","HC01_VC77","HC01_VC78","HC01_VC79","HC01_VC80","HC01_VC81","HC01_VC82","HC01_VC83","HC01_VC84"]
new_param = ["ZIP","M_B_S_ARTS_OCCU","SER_OCCU","SALES_OFFICE_OCCU","NATRES_CONST_MAINT_OCCU","PROD_TRANSP_MATMOVING_OCCU","HC01_VC75","HC01_VC76","HC01_VC77","HC01_VC78","HC01_VC79","INCOME_50_75","INCOME_75_99","INCOME_100_150","INCOME_150_200","INCOME_200_"]
print(len(param))
print(len(new_param))
cali_employ_income2 = new_select(cali_employ_income,param,new_param)
cali_employ_income2.columns = new_param


# In[267]:


cali_employ_income2.dtypes


# In[268]:


cali_employ_income2["INCOME_0_49"] = cali_employ_income2["HC01_VC75"] + cali_employ_income2["HC01_VC76"] + cali_employ_income2["HC01_VC77"] + cali_employ_income2["HC01_VC78"] + cali_employ_income2["HC01_VC79"]


# In[269]:


cali_employ_income2.drop(["HC01_VC75","HC01_VC76","HC01_VC77","HC01_VC78","HC01_VC79"], axis = 1, inplace = True)


# In[270]:


cd5 = new_merge(cd4,cali_employ_income2)


# In[271]:


cd5


# In[272]:


cd5.dtypes


# In[273]:


cali_housing = pd.read_csv("State_wise_data/california/aff_download-7/ACS_17_5YR_DP04_with_ann.csv")


# In[274]:


cali_housing = cali_housing.iloc[1:]


# In[275]:


cali_housing["GEO.id2"] = cali_housing["GEO.id2"].astype("int") 


# In[276]:


param = ["GEO.id2","HC01_VC14","HC01_VC15","HC01_VC16","HC01_VC17","HC01_VC18","HC01_VC19","HC01_VC20","HC01_VC65","HC01_VC66","HC01_VC86","HC01_VC87","HC01_VC88","HC01_VC99","HC01_VC97","HC01_VC101","HC01_VC120","HC01_VC121","HC01_VC122","HC01_VC123","HC01_VC124","HC01_VC125","HC01_VC126","HC01_VC127","HC01_VC184","HC01_VC185","HC01_VC186","HC01_VC187","HC01_VC188","HC01_VC189","HC01_VC190"]
new_param = ["ZIP","1UNIT_DET","1UNIT_ATT","HC01_VC16","HC01_VC17","HC01_VC18","HC01_VC19","HC01_VC20","OWNER_OCCU","RENT_OCCU","OWNER_OCCU_1_VEH","OWNER_OCCU_2_VEH","OWNER_OCCU_3_","SOLAR_HEATING_HOUSE","COAL_HEATING_HOUSE","NOFUEL_HEATING_HOUSE","HC01_VC120","HC01_VC121","HC01_VC122","HC01_VC123","HC01_VC124","HC01_VC125","HC01_VC126","HC01_VC127","HC01_VC184","HC01_VC185","HC01_VC186","HC01_VC187","HC01_VC188","HC01_VC189","HC01_VC190"]
cali_housing2 = new_select(cali_housing,param,new_param)
cali_housing2.columns = new_param


# In[277]:


cali_housing2["2_4UNIT"] = cali_housing2['HC01_VC16'] + cali_housing2['HC01_VC17']
cali_housing2.drop(["HC01_VC16","HC01_VC17"], axis = 1, inplace = True)


# In[278]:


'''def column_add(self,column,new_name):
#how to pass the same dataframe to a function.. understand pass by value, pass by reference'''


# In[279]:


cali_housing2["5_19UNIT"] = cali_housing2["HC01_VC18"] + cali_housing2["HC01_VC19"]
cali_housing2.drop(["HC01_VC18","HC01_VC19"], axis = 1, inplace = True)


# In[280]:


cali_housing2["20UNIT"] = cali_housing2["HC01_VC20"]
cali_housing2.drop(["HC01_VC20"], axis = 1, inplace = True)


# In[281]:


cali_housing2["HV_50_150"] = cali_housing2["HC01_VC120"] + cali_housing2["HC01_VC121"] + cali_housing2["HC01_VC122"]
cali_housing2.drop(["HC01_VC120","HC01_VC121","HC01_VC122"], axis = 1, inplace = True)


# In[282]:


cali_housing2["HV_150_500"] = cali_housing2["HC01_VC123"] + cali_housing2["HC01_VC124"] + cali_housing2["HC01_VC125"]
cali_housing2.drop(["HC01_VC123","HC01_VC124","HC01_VC125"], axis = 1, inplace = True)


# In[283]:


cali_housing2["HV_500_1000"] = cali_housing2["HC01_VC126"]
cali_housing2.drop(["HC01_VC126"], axis = 1, inplace = True)


# In[284]:


cali_housing2["HV_1000"] = cali_housing2["HC01_VC127"]
cali_housing2.drop(["HC01_VC127"], axis = 1, inplace = True)


# In[285]:


cali_housing2


# In[286]:


cali_housing2["RENT_500_1500"] = cali_housing2["HC01_VC184"] + cali_housing2["HC01_VC185"] + cali_housing2["HC01_VC186"]
cali_housing2.drop(["HC01_VC184","HC01_VC185","HC01_VC186"], axis = 1, inplace = True)


# In[287]:


cali_housing2["RENT_1500_2500"] = cali_housing2["HC01_VC187"] + cali_housing2["HC01_VC188"] 
cali_housing2.drop(["HC01_VC187","HC01_VC188"], axis = 1, inplace = True)


# In[288]:


cali_housing2["RENT_2500_"] =  cali_housing2["HC01_VC189"] + cali_housing2["HC01_VC190"]
cali_housing2.drop(["HC01_VC189","HC01_VC190"], axis = 1, inplace = True)


# In[289]:


cali_housing2


# In[290]:


cd6 = new_merge(cd5,cali_housing2)


# In[291]:


cd6


# In[292]:


cd6.columns


# In[293]:


cd6["ZIP"] = cd6["ZIP"].astype("int")


# In[294]:


cd7 = cd6


# In[301]:


cd7.to_csv("cali_augment.csv")


# In[302]:


cd7


# In[303]:


level_ultimate = level1.merge(cd7, on = "ZIP", how = "inner")


# In[412]:


level1.head()


# In[413]:


level_ultimate.head()


# In[327]:


level_ultimate.columns


# In[332]:


california_ev_zip = california_ev_zip.reset_index()


# In[342]:


level_ultimate = level_ultimate.merge(california_ev_zip, on = "ZIP", how = "left")


# In[345]:


level_ultimate.fillna(0,inplace = True)


# In[348]:


level_ultimate_copy = level_ultimate.copy()


# In[352]:


level_ultimate_copy.iloc[:,43:52]


# In[357]:


drop = level_ultimate_copy.iloc[:,43:52].columns.tolist()


# In[359]:


drop.append("ZIP")


# In[360]:


level_ultimate_copy.drop(drop, axis = 1, inplace = True)


# In[390]:


level_5 = level_ultimate_copy.copy()
level_5


# In[394]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
#% matplotlib inline

import numpy as np
from sklearn.cluster import KMeans


km = KMeans(n_clusters=5)
level_5["clusters"] = km.fit_predict(level_5)


# In[395]:


reduced_data = PCA(n_components=2).fit_transform(backup)
results = pd.DataFrame(reduced_data,columns=['pca1','pca2'])

sns.scatterplot(x="pca1", y="pca2", hue=backup['clusters'], data=results)
plt.title('K-means Clustering with 2 dimensions')
plt.show()


# In[396]:


Sum_of_squared_distances = []
K = range(1,10)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(new)
    Sum_of_squared_distances.append(km.inertia_)
    
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# In[399]:


level_5["clusters"].value_counts()


# In[410]:


final_df = pd.concat([level_5,level_ultimate["ZIP"]], axis = 1)


# In[411]:


final_df

