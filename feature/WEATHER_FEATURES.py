# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 16:59:42 2017

@author: Flamingo
"""

import numpy as np
import pandas as pd
import math
import sys
sys.path.append('../TOOLS')
from IJCAI2017_TOOL import *

def SSD(Temp,Velo,Humi):
    score = (1.818*Temp+18.18) * (0.88+0.002*Humi) + 1.0*(Temp -32)/(45-Temp) - 3.2*Velo  + 18.2
    return score
    
WEATHER_raw = pd.read_csv('../additional/WEATHER_raw.csv',encoding = 'gbk',low_memory=False)
#%%

def AMPM2decimal(ser):
    tt = ser.replace(' ',':').split(':')
    tt[0] = np.int(tt[0])%12 
    if (tt[2] == 'AM'):
        return np.float(tt[0]) + np.float(tt[1])/60.
    if (tt[2] == 'PM'):
        return np.float(tt[0]) + np.float(tt[1])/60. + 12.
        
def Eventclean(ser):
    try:
        if (math.isnan(ser)):
            return 'None'
    except:
        tt = ser.replace('\n','\r').replace('\t','\r').split('\r')
        tt2 = ''.join(tt)
        return tt2


    
#%% clean the raw data
WEATHER_raw =   WEATHER_raw[['DATE','Time','Temp','Visibility','Wind_speed','Humidity','Event','Condition','CITY_EN']]

WEATHER_raw['Time']  = [(lambda x:AMPM2decimal(x) ) (x) for x in  WEATHER_raw['Time']]
WEATHER_raw['Event'] = [(lambda x:Eventclean(x) ) (x) for x in  WEATHER_raw['Event']]
WEATHER_raw['Visibility'] = WEATHER_raw['Visibility'].replace('-',np.nan).fillna(method='ffill')
WEATHER_raw['Visibility'] = pd.to_numeric(WEATHER_raw['Visibility'], errors='ignore')

WEATHER_raw['Temp'] = WEATHER_raw['Temp'].replace('-',0.0)
WEATHER_raw['Temp'] = pd.to_numeric(WEATHER_raw['Temp'], errors='ignore')

WEATHER_raw.loc[ WEATHER_raw['Wind_speed'] == 'Calm','Wind_speed']= 0.0
WEATHER_raw['Wind_speed'] = WEATHER_raw['Wind_speed'].replace('-','3.6')
WEATHER_raw['Wind_speed'] = pd.to_numeric(WEATHER_raw['Wind_speed'], errors='ignore')
WEATHER_raw['Wind_speed'] = WEATHER_raw['Wind_speed']/3.6

WEATHER_raw['Humidity'] = WEATHER_raw['Humidity'].replace('N/A%','5%')
WEATHER_raw.loc[ WEATHER_raw['Humidity'] == '%','Humidity']= '5%'
WEATHER_raw['Humidity'] = [(lambda x: (np.int(x.split('%')[0]) ) ) (x) for x in WEATHER_raw['Humidity']]

WEATHER_raw['SSD'] = SSD(WEATHER_raw['Temp'] ,WEATHER_raw['Wind_speed'],WEATHER_raw['Humidity'])

WEATHER_raw.loc[ WEATHER_raw['Condition'] == 'Unknown','Condition']= np.nan
WEATHER_raw['Condition'] = WEATHER_raw['Condition'].fillna(method='ffill')


WEATHER_CON_LEVEL = pd.read_csv('WEATHER_CON_LEVEL.csv')
WEATHER_raw = pd.merge(WEATHER_raw, WEATHER_CON_LEVEL, on = 'Condition', how = 'left')
WEATHER_raw[['RAIN_IND','CLEAR_IND']] = WEATHER_raw[['RAIN_IND','CLEAR_IND']].fillna(0.0)


WEATHER_raw = WEATHER_raw[['DATE','Time','CITY_EN','SSD','RAIN_IND','CLEAR_IND']]


time1 = WEATHER_raw[((WEATHER_raw['Time']<=18.5) & ((WEATHER_raw['Time']>=11)) )]
#
time1_group = time1.groupby(['CITY_EN','DATE'],as_index = False).mean()
#
time1_group['SSD_C'] = np.abs(time1_group['SSD']-60) - np.abs(time1_group['SSD'].shift(1) -60)


time1_group = time1_group[((time1_group['DATE']<='2016-11-20') &(time1_group['DATE']>='2015-06-26')) ]


time1_group = time1_group.rename(columns = {'SSD':'RC','SSD_C':'RE','RAIN_IND':'RG','CLEAR_IND':'RI'})
#
time1_group = time1_group[['CITY_EN','DATE','RC','RE','RG','RI']]
time1_group.to_csv('WEATHER_FEATURES.csv',index = False)