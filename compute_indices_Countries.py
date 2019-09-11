# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:31:05 2019

@author: guillaume
"""
import xarray as xr
from osgeo import ogr
import numpy as np
import geopandas as gpd


pr_in = 'K:/DATA/REANALYSES/ERA5/PrecTOT/' 
t_in = 'K:/DATA/REANALYSES/ERA5/Mean_T2m/' 
file_mask = 'D:/Utilisateurs/guillaume/Documents/GitHub/InSIGHT-PHAC/Mask/' 

multi_tmin = t_in + 'Monthly_Mean_t2m_*.nc'  

ds = xr.open_mfdataset(multi_tmin)
ds_date_range = ds.sel(time=slice('1985', '2014'))
climatology = ds_date_range.groupby('time.month').mean('time')

ds_year = ds.sel(time='2016')
anomalies = ds_year.groupby('time.month') - climatology
       
        
shapes = gpd.read_file("D:/Utilisateurs/guillaume/Documents/GitHub/InSIGHT-PHAC/Countries/Countries_Final-polygon.shp")
list(shapes.columns.values)
for name in shapes['NAME']:
    mask = np.load(file_mask + 'mask_'+str(name.replace(' ','_'))+'.npy')
    np.max(mask)
    anomalies_mask = anomalies.where(mask == 1)
    climatology_mask = climatology.where(mask == 1)
    df=anomalies_mask.t2m.mean(dim=('longitude','latitude')).to_dataframe()
           
        





