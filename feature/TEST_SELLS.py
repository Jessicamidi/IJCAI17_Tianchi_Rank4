# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 20:50:03 2017

@author: Flamingo
"""
import pandas as pd
import numpy as np
import datetime
import copy

import sys
sys.path.append('../TOOLS')
from IJCAI2017_TOOL import *

PAYNW = pd.read_csv('../data_new/user_pay_new.csv')

PAYNW_TAB = pd.pivot_table(PAYNW, values=['Num_post'], index=['SHOP_ID'],columns=['DATE'], aggfunc=np.sum)   
PAYNW_TAB = pd.concat( [PAYNW_TAB[PAYNW_TAB.columns[0:169:1]], pd.DataFrame({'A':[np.nan],},index=np.arange(1,2001)),PAYNW_TAB[PAYNW_TAB.columns[169::1]] ], axis = 1)
PAYNW_TAB.columns = [str((datetime.datetime.strptime('20150626','%Y%m%d') + datetime.timedelta(days=x)).date()) for x in range( PAYNW_TAB.shape[1])] 
inspect_cols = [str((datetime.datetime.strptime('20161009','%Y%m%d') + datetime.timedelta(days=x)).date()) for x in range( 23)] 

PAYNW_TAB_OCT = PAYNW_TAB.loc[:,'2016-10-09':'2016-10-31']
PAYNW_TAB_OCT.reset_index(level=0, inplace=True)

SHOP_MELT = pd.melt(PAYNW_TAB_OCT , id_vars=['SHOP_ID'], value_vars = inspect_cols)
SHOP_MELT = SHOP_MELT.rename(columns = {'variable':'DATE'})

#%% find all the shops with small value, substitude with the mininum value of day of week in 4 weeks
SMALL_SHOP = SHOP_MELT.loc[SHOP_MELT['value'] <= 10,:]
SMALL_SHOP = SMALL_SHOP.sort_values(by = ['SHOP_ID','DATE'])
SMALL_count = SMALL_SHOP.groupby(by = ['SHOP_ID'],as_index = False).count()
SMALL_count = SMALL_count[SMALL_count['DATE']<=2]
SMALL_SHOP = SMALL_SHOP[SMALL_SHOP.SHOP_ID.isin(SMALL_count.SHOP_ID)]
SMALL_SHOP.index = np.arange(len(SMALL_SHOP))
Substitude_list = []
for ind,value in SMALL_SHOP.iterrows():
    SHOP_ID = value.SHOP_ID
    DATE  = value.DATE   
    Day_shift_list = [-14,-7,7,14]
    Shop_sub_list = []
    for shift_ind in Day_shift_list:   
        DATE_shift = str((datetime.datetime.strptime(DATE,'%Y-%m-%d') + datetime.timedelta(days = shift_ind)).date()) 
        try:
            value_append = PAYNW_TAB.loc[SHOP_ID, DATE_shift ]
        except:
            value_append = np.nan
        Shop_sub_list.append(value_append)
    Substitude_list.append(Shop_sub_list)

Substitude_list = pd.DataFrame( Substitude_list )
SMALL_SHOP = pd.concat([SMALL_SHOP, Substitude_list] ,axis = 1)
SMALL_SHOP['Num_post'] = SMALL_SHOP[np.arange(4)].min(axis = 1)

for ind,value in SMALL_SHOP.iterrows():
    SHOP_ID = value.SHOP_ID
    DATE = value.DATE
    PAYNW_TAB.loc[SHOP_ID,DATE] = value.Num_post
    

#%% substitude with fill oct

 
PAYNW_TAB_FIX = pd.read_csv('../data_new/FillOct.csv')
PAYNW_TAB_FIX['DATE'] = [ (lambda x:str(datetime.datetime.strptime(x,'%Y/%m/%d').date() ) ) (x) for x in  PAYNW_TAB_FIX['DATE']]

for ind,value in PAYNW_TAB_FIX.iterrows():
    SHOP_ID = value.SHOP_ID
    DATE = value.DATE
    PAYNW_TAB.loc[SHOP_ID,DATE] = value.Num_post
    
    

TRN_N = 21
TST_N = 14

TEST= pd.DataFrame()
TRN_END = datetime.datetime.strptime('2016-10-31','%Y-%m-%d')
TRN_STA = (TRN_END - datetime.timedelta(days=(TRN_N-1)) )
TST_STA = (TRN_END +  datetime.timedelta(days=(1)) )
TST_END = (TRN_END +  datetime.timedelta(days=(TST_N)) )
test_date_zip = zip([str(TRN_STA.date())],[str(TRN_END.date())],[str(TST_STA.date())],  [str(TST_END.date()) ])
TEST =  PAYNW_TAB.loc[:,str(TRN_STA.date()):str(TRN_END.date())]   
TEST.reset_index(level=0, inplace=True)
end_date = datetime.datetime.strptime('2016-10-31','%Y-%m-%d')
TEST.loc[:,'TRN_STA']  = str(TRN_STA.date())
TEST.loc[:,'TRN_END']  = str(TRN_END.date())
TEST.loc[:,'TST_STA']  = str(TST_STA.date())
TEST.loc[:,'TST_END']  = str(TST_END.date())
TEST_TRN_C =  map(lambda x:'SA'+ str(x).zfill(2), np.arange(TRN_N))
TEST.columns = ['SHOP_ID']  + TEST_TRN_C  + ['TRN_STA','TRN_END','TST_STA','TST_END']

#%%

TEST.to_csv('TEST_SELLS.csv',index = False)
   