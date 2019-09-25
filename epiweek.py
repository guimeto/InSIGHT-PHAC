# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 12:12:33 2019

@author: guillaume
"""
import pandas as pd
from epiweeks import Week, Year
df_epi=[]
for week in Year(2015).iterweeks():
    day = week.week
    day_ini = week.startdate().strftime("%Y-%m-%d")
    day_end = week.enddate().strftime("%Y-%m-%d")
    data = {'Week':[day],
            'datei':[day_ini],
            'datef':[day_end]}
    df=pd.DataFrame(data)
    df_epi.append(pd.DataFrame(data).set_index('Week'))
    
df_epi = pd.concat(df_epi, axis=0).sort_index()    