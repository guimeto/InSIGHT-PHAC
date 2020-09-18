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
from epiweeks import Week, Year
file_mask = 'D:/Utilisateurs/guillaume/Documents/GitHub/InSIGHT-PHAC/Mask/' 
pr_in = 'K:/DATA/REANALYSES/ERA5/PR/'

data = pr_in + 'ERA5_PR_*.nc4'
ds = xr.open_mfdataset(data)
        
ds = ds * 1000  # convert from meter to mm

year_to_study = 2015 
# compute epiweek for 2015 and 2016
df_epi=[]
for week in Year(year_to_study).iterweeks():
    day = week.week
    day_ini = week.startdate().strftime("%Y-%m-%d")
    day_end = week.enddate().strftime("%Y-%m-%d")
    data = {'Week':[day],
            'datei':[day_ini],
            'datef':[day_end]}
    df=pd.DataFrame(data)
    df_epi.append(pd.DataFrame(data).set_index('Week'))
    
df_epi = pd.concat(df_epi, axis=0).sort_index()    

# compute climatologie bi-epiweek : 26 * 2 epiweeks
iw1 = 0
iw2 = 1
climatologie_epiweek = []
while iw2  <  df_epi.shape[0] : 
    datasets = []
    for year in range(1985,2015):
        
        datei1 = '-'.join((str(year),df_epi.iloc[iw1,0].split('-')[1],df_epi.iloc[iw1,0].split('-')[2]))
        datef1 = '-'.join((str(year),df_epi.iloc[iw1,1].split('-')[1],df_epi.iloc[iw1,1].split('-')[2]))
        ds_week1 = ds.sel(time=slice(datei1, datef1)).sum('time')   
        if iw2 == 51:
            datei2 = '-'.join((str(year),df_epi.iloc[iw2,0].split('-')[1],df_epi.iloc[iw2,0].split('-')[2]))
            datef2 = '-'.join((str(year+1),df_epi.iloc[iw2,1].split('-')[1],df_epi.iloc[iw2,1].split('-')[2]))
        else:
            datei2 = '-'.join((str(year),df_epi.iloc[iw2,0].split('-')[1],df_epi.iloc[iw2,0].split('-')[2]))
            datef2 = '-'.join((str(year),df_epi.iloc[iw2,1].split('-')[1],df_epi.iloc[iw2,1].split('-')[2]))
        ds_week2 = ds.sel(time=slice(datei2, datef2)).sum('time')  
        ds_new = xr.concat([ds_week1, ds_week2])  
        datasets.append(ds_new)
        
    climatologie_epiweek = (xr.concat(datasets).mean('concat_dims'))  
    
    datei1 = '-'.join((str(year_to_study),df_epi.iloc[iw1,0].split('-')[1],df_epi.iloc[iw1,0].split('-')[2]))
    datef1 = '-'.join((str(year_to_study),df_epi.iloc[iw1,1].split('-')[1],df_epi.iloc[iw1,1].split('-')[2]))
    ds_week1 = ds.sel(time=slice(datei1, datef1)).sum('time')  
    if iw2 == 51:
        datei2 = '-'.join((str(year_to_study),df_epi.iloc[iw2,0].split('-')[1],df_epi.iloc[iw2,0].split('-')[2]))
        datef2 = '-'.join((str(year_to_study+1),df_epi.iloc[iw2,1].split('-')[1],df_epi.iloc[iw2,1].split('-')[2]))
    else:
        datei2 = '-'.join((str(year_to_study),df_epi.iloc[iw2,0].split('-')[1],df_epi.iloc[iw2,0].split('-')[2]))
        datef2 = '-'.join((str(year_to_study),df_epi.iloc[iw2,1].split('-')[1],df_epi.iloc[iw2,1].split('-')[2]))
        
    ds_week2 = ds.sel(time=slice(datei2, datef2)).sum('time')
    ds_new = ds_week1 + ds_week2
    
    anomalies = ((ds_new / climatologie_epiweek)-1)*100

    shapes = gpd.read_file("D:/Utilisateurs/guillaume/Documents/GitHub/InSIGHT-PHAC/Zones/Masque.shp")
    
    list(shapes.columns.values)
    # copy poly to new GeoDataFrame
    points = shapes.copy()
    # change the geometry
    points.geometry = points['geometry'].centroid
    # same crs
    points.crs =shapes.crs
    points.head()
    for i in range(0,len(shapes)):
        shapes.loc[i,'centroid_lon'] = shapes.geometry.centroid.x.iloc[i]
        shapes.loc[i,'centroid_lat'] = shapes.geometry.centroid.y.iloc[i]

    df_ano=[]
    df_clim=[]
    
    for name in shapes['NAME']:
        mask = np.load(file_mask + 'mask_'+str(name.replace(' ','_'))+'.npy')
        print(np.max(mask))
        if np.max(mask) == 0 :
            print('Recherche du centroid')
            lati =shapes.loc[shapes['NAME'] == name]['centroid_lat'].values
            loni =shapes.loc[shapes['NAME'] == name]['centroid_lon'].values + 360
            
            anomalies_point = anomalies.tp(longitude=float(loni)  , latitude=float(lati) , method='nearest')
            climatology_point = climatologie_epiweek.sel(longitude=float(loni)  , latitude=float(lati)  , method='nearest')        
            columns = ['longitude', 'latitude']
            anomalies_point.tp.values
            
            
            df_clim.append(climatology_point.tp.to_dataframe())
            df_ano.append(anomalies_point.tp.to_dataframe().set_index('month'))
        else:
            print('Application du masque np.array')
           
            anomalies_mask = anomalies.where(mask == 1)
            
            
            anomalies_mask.to_netcdf('mask.nc')
            lat_bnd = [35, 0]
            lon_bnd = [240, 280]
            anomalies_mask.tp.sel(longitude=slice(*lon_bnd), latitude=slice(*lat_bnd),)
            anomalies_mask.sel( )
            climatology_mask = climatologie_epiweek.where(mask == 1)
            df_ano.append(anomalies_mask.tp.mean(dim=('longitude','latitude')).to_dataframe())      
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
ax.set_title('Anomalies relatives des accumulations epiweek de précipitation de 2015 par rapport à la normale 1985-2014', weight='bold', fontsize="x-large")
figure = ax.get_figure()    
figure.set_size_inches(22, 15) 
plt.savefig("Anomalies_Bi-week_Precipitation_2015_vs_1985-2014_new.png", bbox_inches="tight")
plt.close()
ax = plt.axes()
sns.heatmap(df_clim, cmap='coolwarm', linewidths=0.5, annot=True , ax = ax,vmin=0, vmax=150, fmt='.0f',yticklabels=True, cbar_kws={'label': 'mm'})
ax.set_title('Climatologie biweek des accumulations mensuelles des précipitations (1985-2014)', weight='bold', fontsize="x-large")
figure = ax.get_figure()    
figure.set_size_inches(22, 15) 
plt.savefig("Climatologies_biweek_Precipitation_1985-2014_new.png", bbox_inches="tight")

df_ano.to_csv("Anomalies_biweek_Precipitation_2015_vs_1985-2014_newe.csv",  header = True, sep = ',')
df_clim.to_csv("Climatologies_Precipitation_biweek_1985-2014_new.csv", header = True, sep = ',')




