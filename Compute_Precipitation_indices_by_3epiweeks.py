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
  
lat_bnd = [62, -70]
lon_bnd = [230, 340]


Imp_Lats =  ds.sel(longitude=slice(*lon_bnd), latitude=slice(*lat_bnd),)['latitude'].values
Imp_Lons =  ds.sel(longitude=slice(*lon_bnd), latitude=slice(*lat_bnd),)['longitude'].values
lon2d, lat2d = np.meshgrid(Imp_Lons, Imp_Lats)

# compute climatologie bi-epiweek : 26 * 2 epiweeks
iw1 = 0
iw2 = 1
iw3 = 2

climatologie_epiweek = []
while iw2  <  df_epi.shape[0] : 
    datasets = []
    for year in range(1985,2015):
        datei1 = '-'.join((str(year),df_epi.iloc[iw1,0].split('-')[1],df_epi.iloc[iw1,0].split('-')[2]))
        datef1 = '-'.join((str(year),df_epi.iloc[iw1,1].split('-')[1],df_epi.iloc[iw1,1].split('-')[2]))
        ds_week1 = ds.sel(longitude=slice(*lon_bnd), latitude=slice(*lat_bnd),time=slice(datei1, datef1)).sum('time')
        
        datei2 = '-'.join((str(year),df_epi.iloc[iw2,0].split('-')[1],df_epi.iloc[iw2,0].split('-')[2]))
        datef2 = '-'.join((str(year),df_epi.iloc[iw2,1].split('-')[1],df_epi.iloc[iw2,1].split('-')[2]))
        ds_week2 = ds.sel(longitude=slice(*lon_bnd), latitude=slice(*lat_bnd),time=slice(datei2, datef2)).sum('time')
        
        datei3 = '-'.join((str(year),df_epi.iloc[iw3,0].split('-')[1],df_epi.iloc[iw3,0].split('-')[2]))
        datef3 = '-'.join((str(year),df_epi.iloc[iw3,1].split('-')[1],df_epi.iloc[iw3,1].split('-')[2]))
        ds_week3 = ds.sel(longitude=slice(*lon_bnd), latitude=slice(*lat_bnd),time=slice(datei3, datef3)).sum('time')   
        
        ds_new = (ds_week1 + ds_week2 + ds_week3)    
        datasets.append(ds_new.tp.values)
    climatologie_epiweek.append(np.mean(datasets,axis=0))
  #   climatologie_epiweek.append(xr.concat(datasets).mean('concat_dims'))        
    iw1+=3
    iw2+=3
    iw3+=3

# compute year to sutudy
iw1 = 0
iw2 = 1
iw3 = 2

dataset_year = []
while iw2  <  df_epi.shape[0] : 
    
    datei1 = '-'.join((str(year),df_epi.iloc[iw1,0].split('-')[1],df_epi.iloc[iw1,0].split('-')[2]))
    datef1 = '-'.join((str(year),df_epi.iloc[iw1,1].split('-')[1],df_epi.iloc[iw1,1].split('-')[2]))
    ds_week1 = ds.sel(longitude=slice(*lon_bnd), latitude=slice(*lat_bnd),time=slice(datei1, datef1)).sum('time')
        
    datei2 = '-'.join((str(year),df_epi.iloc[iw2,0].split('-')[1],df_epi.iloc[iw2,0].split('-')[2]))
    datef2 = '-'.join((str(year),df_epi.iloc[iw2,1].split('-')[1],df_epi.iloc[iw2,1].split('-')[2]))
    ds_week2 = ds.sel(longitude=slice(*lon_bnd), latitude=slice(*lat_bnd),time=slice(datei2, datef2)).sum('time')
        
    datei3 = '-'.join((str(year),df_epi.iloc[iw3,0].split('-')[1],df_epi.iloc[iw3,0].split('-')[2]))
    datef3 = '-'.join((str(year),df_epi.iloc[iw3,1].split('-')[1],df_epi.iloc[iw3,1].split('-')[2]))
    ds_week3 = ds.sel(longitude=slice(*lon_bnd), latitude=slice(*lat_bnd),time=slice(datei3, datef3)).sum('time')    
        
    ds_new = (ds_week1 + ds_week2 + ds_week3)     
    dataset_year.append(pd.DataFrame(ds_new.tp.values))
    
    iw1+=3
    iw2+=3
    iw3+=3
    
anomalies = []
for epi in range(0,len(climatologie_epiweek)): 
    anomalies.append(((dataset_year[int(epi)]/climatologie_epiweek[int(epi)])-1)*100)
   
shapes = gpd.read_file("D:/Utilisateurs/guillaume/Documents/GitHub/InSIGHT-PHAC/Zones/Masque.shp")

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


def getclosest_ij(lats,lons,latpt,lonpt):
     # find squared distance of every point on grid
     dist_sq = (lats-latpt)**2 + (lons-lonpt)**2 
     # 1D index of minimum dist_sq element
     minindex_flattened = dist_sq.argmin()
     # Get 2D index for latvals and lonvals arrays from 1D index
     return np.unravel_index(minindex_flattened, lats.shape)
 
    
# save the shapefile
points.to_file('geoch_centroid.shp')

dataframe_anomalies=[]
dataframe_climatologies=[]
for name in shapes['NAME']:
    mask = np.load(file_mask + 'mask_'+str(name.replace(' ','_'))+'_small.npy')
    print(np.max(mask))
    df_ano=[]
    df_clim=[]
    if np.max(mask) == 0 :
        print('Recherche du centroid')
        lati =shapes.loc[shapes['NAME'] == name]['centroid_lat'].values
        loni =shapes.loc[shapes['NAME'] == name]['centroid_lon'].values + 360
        
        for t in range(0,len(anomalies)):
            iy_min, ix_min = getclosest_ij(lat2d, lon2d, lati, loni)          
            anomalies_point = anomalies[t].iloc[iy_min, ix_min]           
            climatology_point = climatologie_epiweek[t][iy_min, ix_min]  
            

             
            df_ano.append(anomalies_point)
            df_clim.append(climatology_point)
    else:
        print('Application du masque np.array')
        for t in range(0,len(anomalies)):
            tmp_ano = anomalies[t].values[mask==1]
            tmp_clim = climatologie_epiweek[t][mask==1]


                
            df_ano.append(np.mean(tmp_ano))      
            df_clim.append(np.mean(tmp_clim))
    if shapes['NAME'][0] == name:
        dataframe_anomalies = pd.DataFrame(df_ano,columns=[name])
        dataframe_climatologies = pd.DataFrame(df_clim,columns=[name])
    else:
        dataframe_anomalies[name]= pd.DataFrame(df_ano)
        dataframe_climatologies[name]= pd.DataFrame(df_clim)       


#dataframe_anomalies[dataframe_anomalies > 1000] = dataframe_anomalies/10
 
list_tmp = []
for t in range(0,len(anomalies)):
    list_tmp.append('W %d + W %d + W %d' %(df_epi.index[::3][t],df_epi.index[1::3][t],df_epi.index[2::3][t]))

dataframe_anomalies['Epiweeks'] = pd.DataFrame(list_tmp)
dataframe_anomalies = dataframe_anomalies.set_index('Epiweeks')  
dataframe_climatologies['Epiweeks'] = pd.DataFrame(list_tmp)
dataframe_climatologies = dataframe_climatologies.set_index('Epiweeks')  

ax = plt.axes()
sns.heatmap(dataframe_anomalies, cmap='coolwarm', linewidths=0.5, annot=True , ax = ax,vmin=-100.1, vmax=100.1,center=0, fmt='.0f',yticklabels=True, cbar_kws={'label': '%'})
ax.set_title('Anomalies relatives des accumulations sur 3 semaines epiweeks: '+str(year_to_study)+' par rapport à la normale 1985-2014', weight='bold', fontsize="x-large")
figure = ax.get_figure()    
figure.set_size_inches(22, 15) 
plt.savefig("Anomalies_tri-week_Precipitation_"+str(year_to_study)+"_vs_1985-2014_new.png", bbox_inches="tight")
plt.close()

ax = plt.axes()
sns.heatmap(dataframe_climatologies, cmap='coolwarm', linewidths=0.5, annot=True , ax = ax,vmin=0, vmax=150, fmt='.0f',yticklabels=True, cbar_kws={'label': 'mm'})
ax.set_title('Climatologie des accumulations sur 3 epiweek des précipitations (1985-2014)', weight='bold', fontsize="x-large")
figure = ax.get_figure()    
figure.set_size_inches(22, 15) 
plt.savefig("Climatologies_triweek_Precipitation_1985-2014_new.png", bbox_inches="tight")

dataframe_anomalies.to_csv("Anomalies_triweek_Precipitation_"+str(year_to_study)+"_vs_1985-2014_new.csv",  header = True, sep = ',')
dataframe_climatologies.to_csv("Climatologies_Precipitation_triweek_1985-2014_new.csv", header = True, sep = ',')




