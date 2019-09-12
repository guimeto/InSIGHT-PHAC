# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:31:05 2019

@author: guillaume
"""
from matplotlib import pyplot as plt
import xarray as xr
from osgeo import ogr
import numpy as np
import geopandas as gpd
import pandas as pd
import seaborn as sns 

pr_in = 'K:/DATA/REANALYSES/ERA5/PrecTOT/' 

file_mask = 'D:/Utilisateurs/guillaume/Documents/GitHub/InSIGHT-PHAC/Mask/' 

multi_pr = pr_in + 'Monthly_Total_PR_*.nc'  

ds = xr.open_mfdataset(multi_pr)
ds_date_range = ds.sel(time=slice('1985', '2014'))
climatology = ds_date_range.groupby('time.month').mean('time')

ds_year = ds.sel(time='2016')
anomalies = ((ds_year.groupby('time.month') / climatology)-1)*100
       
        
shapes = gpd.read_file("D:/Utilisateurs/guillaume/Documents/GitHub/InSIGHT-PHAC/Countries/Countries_Final-polygon.shp")
list(shapes.columns.values)
# copy poly to new GeoDataFrame
points = shapes.copy()
# change the geometry
points.geometry = points['geometry'].centroid
# same crs
points.crs =shapes.crs
points.head()

plt.rcParams["figure.figsize"]=[16,9]
base = shapes.plot(color='white', edgecolor='black')
points.plot(ax=base, marker='o', color='red', markersize=10)
plt.xlim(-80,-50)
plt.ylim(0,20)
plt.savefig("centroids.png", bbox_inches="tight")

for i in range(0,len(shapes)):
    shapes.loc[i,'centroid_lon'] = shapes.geometry.centroid.x.iloc[i]
    shapes.loc[i,'centroid_lat'] = shapes.geometry.centroid.y.iloc[i]



# save the shapefile
points.to_file('geoch_centroid.shp')
df_ano=[]
df_clim=[]

for name in shapes['NAME']:
    mask = np.load(file_mask + 'mask_'+str(name.replace(' ','_'))+'.npy')
    print(np.max(mask))
    if np.max(mask) == 0 :
        print('Recherche du centroid')
        lati =shapes.loc[shapes['NAME'] == name]['centroid_lat'].values
        loni =shapes.loc[shapes['NAME'] == name]['centroid_lon'].values + 360
      
        anomalies_point = anomalies.sel(longitude=float(loni)  , latitude=float(lati)  , method='nearest') 
        climatology_point = climatology.sel(longitude=float(loni)  , latitude=float(lati)  , method='nearest')
        
       
        columns = ['longitude', 'latitude']
        df_clim.append(climatology_point.tp.to_dataframe())
        df_ano.append(anomalies_point.tp.to_dataframe().set_index('month'))
    else:
        print('Application du masque np.array')
        anomalies_mask = anomalies.where(mask == 1)
        climatology_mask = climatology.where(mask == 1)
        df_ano.append(anomalies_mask.tp.mean(dim=('longitude','latitude')).to_dataframe().set_index('month'))
       
        df_clim.append(climatology_mask.tp.mean(dim=('longitude','latitude')).to_dataframe())
df_clim = pd.concat(df_clim, axis=1)           
df_ano = pd.concat(df_ano, axis=1)         
columns = ['longitude', 'latitude']
df_ano = df_ano.drop(columns,axis=1)
df_clim = df_clim.drop(columns,axis=1)
df_ano.columns = shapes['NAME']
df_clim.columns = shapes['NAME']


ax = plt.axes()
sns.heatmap(df_ano, cmap='coolwarm', linewidths=0.5, annot=True , ax = ax,vmin=-100, vmax=100,center=0, fmt='.0f',yticklabels=True, cbar_kws={'label': '%'})
ax.set_title('Anomalies relatives des accumulations mensuelles de précipitation de 2016 par rapport à la normale 1985-2014', weight='bold', fontsize="x-large")
figure = ax.get_figure()    
figure.set_size_inches(22, 15) 
plt.savefig("Anomalies_Mensuelles_Precipitation_2016_vs_1985-2014.png", bbox_inches="tight")
plt.close()
ax = plt.axes()
sns.heatmap(df_clim, cmap='coolwarm', linewidths=0.5, annot=True , ax = ax,vmin=0, vmax=150, fmt='.0f',yticklabels=True, cbar_kws={'label': 'mm'})
ax.set_title('Climatologie des accumulations mensuelles des précipitations (1985-2014)', weight='bold', fontsize="x-large")
figure = ax.get_figure()    
figure.set_size_inches(22, 15) 
plt.savefig("Climatologies_Mensuelles_Precipitation_1985-2014.png", bbox_inches="tight")

df_ano.to_csv("Anomalies_Mensuelles_Precipitation_2016_vs_1985-2014.csv",  header = True, sep = ',')
df_clim.to_csv("Climatologies_Precipitation_Mensuelles_1985-2014.csv", header = True, sep = ',')



