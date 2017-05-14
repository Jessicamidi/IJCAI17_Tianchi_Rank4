# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 23:11:56 2017

@author: aa
"""

import pandas as pd
import numpy as np
import datetime
import copy

import sys
sys.path.append('../TOOLS')
from IJCAI2017_TOOL import *

#%%  readin shop data
HOLI = pd.read_csv('../additional/HOLI.csv')
HOLI = HOLI.set_index(['DATE'],drop = True)
HOLI_TAB = HOLI.transpose()
HOLI_TAB.columns = [str((datetime.datetime.strptime('20150626','%Y%m%d') + datetime.timedelta(days=x)).date()) for x in range( HOLI_TAB.shape[1])] 
#%%  readin shop data
PAYNW = pd.read_csv('../data_new/user_pay_new.csv')
VIENW = pd.read_csv('../data_new/user_view_new.csv')

PAYNW_SHOP_DATE = PAYNW.groupby(['SHOP_ID','DATE'],as_index = False).sum()
PAYNW_SHOP_DATE = PAYNW_SHOP_DATE[['SHOP_ID','DATE','Num_post']]


PAYNW_SHOP_DATE.reset_index(level=0)


PAYNW_TAB = pd.pivot_table(PAYNW_SHOP_DATE, values=['Num_post'], index=['SHOP_ID'],columns=['DATE'], aggfunc=np.sum)   
PAYNW_TAB = pd.concat( [PAYNW_TAB[PAYNW_TAB.columns[0:169:1]], pd.DataFrame({'A':[np.nan],},index=np.arange(1,2001)),PAYNW_TAB[PAYNW_TAB.columns[169::1]] ], axis = 1)
PAYNW_TAB.columns = [str((datetime.datetime.strptime('20150626','%Y%m%d') + datetime.timedelta(days=x)).date()) for x in range( PAYNW_TAB.shape[1])] 
PAYNW_TAB['2015-12-12'] = PAYNW_TAB['2015-12-13']

PAYNW_TAB_T = PAYNW_TAB.transpose()

#%%  shop_related_features
SHOP_INFO = pd.read_csv("SHOP_FEATURES.csv",low_memory=False)
SHOP_SC = ['SC00']
SHOP_SD = map(lambda x:'SD'+ str(x).zfill(2), np.arange(5)) 
SHOP_SE = map(lambda x:'SE'+ str(x).zfill(2), np.arange(1))
SHOP_SF = map(lambda x:'SF'+ str(x).zfill(2), np.arange(1))
SHOP_SG = map(lambda x:'SG'+ str(x).zfill(2), np.arange(4))
SHOP_SH = map(lambda x:'SH'+ str(x).zfill(2), np.arange(2))
SHOP_SI =  [(lambda x:('SI'+ str(x).zfill(2))) (x)  for x in range(10)]
SHOP_SJ = map(lambda x:'SJ'+ str(x).zfill(2), np.arange(15)) 
SHOP_columns = SHOP_SC + SHOP_SD + SHOP_SE + SHOP_SF + SHOP_SG +  SHOP_SH + SHOP_SI + SHOP_SJ

#%%
TRN_N = 21
TST_N = 14   
TST_PAD_N = 14 + 4  

end_date = datetime.datetime.strptime('2016-10-31','%Y-%m-%d')
day_N = 494
date_list = [str((end_date- datetime.timedelta(days=x)).date()) for x in range(day_N)]
date_list.reverse()  

#%%
TRAIN = pd.DataFrame()
train_date_zip = zip(date_list[0:day_N-(TRN_N+TST_N)+1],date_list[TRN_N-1:day_N-TST_N+1],date_list[TRN_N:day_N-TST_N+2],  date_list[TRN_N+TST_N-1:day_N])
train_date_zip_df = pd.DataFrame(train_date_zip)
train_date_zip_df.columns = ['TRN_STA','TRN_END','TST_STA','TST_END']

for TRN_STA,TRN_END,TST_STA,TST_END in train_date_zip:
    TRAIN_temp = PAYNW_TAB.loc[:,TRN_STA:TST_END] 
    TRAIN_temp.columns = np.arange(TRAIN_temp.shape[1])
    TRAIN_temp.reset_index(level=0, inplace=True)
    TRAIN_temp.loc[:,'TRN_STA']  = str(TRN_STA)
    TRAIN_temp.loc[:,'TRN_END'] = str(TRN_END)    
    TRAIN_temp.loc[:,'TST_STA']  = str(TST_STA)
    TRAIN_temp.loc[:,'TST_END'] = str(TST_END)    
    TRAIN = pd.concat( [TRAIN,TRAIN_temp],) 
#%%
TRAIN = TRAIN.reset_index(np.arange(len(TRAIN)),drop = True)
TRAIN_TRN_C =  map(lambda x:'SA'+ str(x).zfill(2), np.arange(TRN_N))
TRAIN_TST_C =  map(lambda x:'SB'+ str(x).zfill(2), np.arange(TST_N))

TRAIN.columns = ['SHOP_ID']  + TRAIN_TRN_C + TRAIN_TST_C  + ['TRN_STA','TRN_END','TST_STA','TST_END']

#%%    
TEST = pd.read_csv('TEST_SELLS.csv')

#%%
result_fix = pd.read_csv('../data_new/basemodel_last6weeks_aver_m105.csv',header=None,names =['SHOP_ID'] + range(14))
result_fmed = pd.DataFrame()
result_fmed['SHOP_ID'] = result_fix['SHOP_ID']
result_fmed['VALUE'] = result_fix.loc[:,np.arange(0,14)].median(axis = 1) 
result_fmed.to_csv('FIXER.csv',index = False)

TRAIN_OK = TRAIN[TRAIN.loc[:,TRAIN_TST_C].isnull().sum(axis = 1)==0]
TRAIN_OK = TRAIN_OK[TRAIN_OK.loc[:,TRAIN_TRN_C].isnull().sum(axis = 1)<=(TRN_N-21)]
TRAIN_OK = pd.merge(TRAIN_OK ,result_fmed,on='SHOP_ID',how = 'left')
TEST = pd.merge(TEST ,result_fmed,on='SHOP_ID',how = 'left')


TRAIN_OK = pd.merge(TRAIN_OK ,SHOP_INFO,on='SHOP_ID',how = 'left')
TEST = pd.merge(TEST ,SHOP_INFO,on='SHOP_ID',how = 'left')

TRAIN_OK['DT'] = pd.to_datetime(TRAIN_OK['TRN_STA']) - pd.to_datetime(TRAIN_OK['SH00'])
TRAIN_OK.loc[:,'DT'] = [(lambda x:(x.days)) (x) for x in TRAIN_OK.loc[:,'DT']]
TRAIN_OK = TRAIN_OK[TRAIN_OK['DT'] >7]
#%%  

TRAIN_OK[TRAIN_TRN_C + TRAIN_TST_C] = np.log(TRAIN_OK[TRAIN_TRN_C + TRAIN_TST_C]).div(np.log(TRAIN_OK['VALUE']),axis=0 )
TEST[TEST_TRN_C] = np.log(TEST[TEST_TRN_C]).div( np.log( TEST['VALUE']),axis=0) 
#%%
HOLI_TRN_CA  = [(lambda x:('NC'+ str(x).zfill(2))) (x)  for x in range(TRN_N)]
HOLI_TST_CA  = [(lambda x:('ND'+ str(x).zfill(2))) (x)  for x in range(-2,TST_PAD_N-2)]

HOLI_TRN = pd.DataFrame()

date_all = copy.deepcopy(train_date_zip)
date_all.append(test_date_zip[0])
date_all_df = pd.DataFrame(date_all)
date_all_df.columns = ['TRN_STA','TRN_END','TST_STA','TST_END']
for TRN_STA,TRN_END,TST_STA,TST_END in date_all:
    tt1 = HOLI_TAB.loc[:,TRN_STA:TRN_END]
    tt1.columns = HOLI_TRN_CA
    
    tst_sta = str((datetime.datetime.strptime(TST_STA,'%Y-%m-%d') + datetime.timedelta(-2)).date())
    tst_end = str((datetime.datetime.strptime(TST_END,'%Y-%m-%d') + datetime.timedelta(2)).date())
    
    tt2 = HOLI_TAB.loc[:,tst_sta :tst_end ]
    tt2.columns = HOLI_TST_CA
    tt3 = pd.concat([tt1,tt2],axis=1)
    
    HOLI_TRN = pd.concat([HOLI_TRN,tt3])
HOLI_TRN = HOLI_TRN.reset_index(np.arange(len(HOLI_TRN)),drop = True)  
HOLI_TRN = HOLI_TRN.join(date_all_df)

#%%
TRAIN_OK = pd.merge(TRAIN_OK ,HOLI_TRN,on=['TRN_STA','TRN_END','TST_STA','TST_END'],how = 'left')
TEST = pd.merge(TEST ,HOLI_TRN,on=['TRN_STA','TRN_END','TST_STA','TST_END'],how = 'left')

#%%
HOLI_RATIO = [(lambda x:('NE'+ str(x).zfill(2))) (x)  for x in range(1)]
def Add_work_ratio(df,HOLI_TRN_CA,HOLI_RATIO ):   
    tt = (df[HOLI_TRN_CA]==0).astype(int).sum(axis=1)
    df[HOLI_RATIO[0]] = tt  
    return df
TRAIN_OK = Add_work_ratio(TRAIN_OK,HOLI_TRN_CA,HOLI_RATIO)
TEST = Add_work_ratio(TEST,HOLI_TRN_CA,HOLI_RATIO)
#%%

PRECIP = pd.read_csv('../additional/PRECIP.csv')
PRECIP_TAB = pd.pivot_table(PRECIP, values=['Precip'], index=['CITY_EN'],columns=['DATE'])
PRECIP_TAB.columns = [str((datetime.datetime.strptime('20150501','%Y%m%d') + datetime.timedelta(days=x)).date()) for x in range( PRECIP_TAB.shape[1])] 
PRECIP_TRN_C = [(lambda x:('RA'+ str(x).zfill(2))) (x)  for x in range(TRN_N)]
PRECIP_TST_C = [(lambda x:('RB'+ str(x).zfill(2))) (x)  for x in range(TST_N)]
PRECIP_TRN = pd.DataFrame()
for TRN_STA,TRN_END,TST_STA,TST_END in date_all: 
    tt1 = PRECIP_TAB.loc[:,TRN_STA:TRN_END ]  
    tt2 = PRECIP_TAB.loc[:,TST_STA:TST_END ]    
    tt3 = pd.concat([tt1,tt2],axis=1)   
    tt3.reset_index(level=0, inplace=True)   
    tt3.columns = ['CITY_EN'] + PRECIP_TRN_C + PRECIP_TST_C
    tt3.loc[:,'TRN_STA'] = TRN_STA
    tt3.loc[:,'TRN_END'] = TRN_END     
    tt3.loc[:,'TST_STA'] = TST_STA
    tt3.loc[:,'TST_END'] = TST_END    
    PRECIP_TRN = pd.concat([PRECIP_TRN,tt3],axis = 0)

PRECIP_TRN = PRECIP_TRN.reset_index(np.arange(len(PRECIP_TRN)),drop = True)  
TRAIN_OK = pd.merge(TRAIN_OK ,PRECIP_TRN,on=['CITY_EN','TRN_STA','TRN_END','TST_STA','TST_END'],how = 'left')
TEST = pd.merge(TEST ,PRECIP_TRN,on=['CITY_EN','TRN_STA','TRN_END','TST_STA','TST_END'],how = 'left')

#%%
WEATHER = pd.read_csv('WEATHER_FEATURES.csv')

WEATHER_TAB = pd.pivot_table(WEATHER, values=['RC','RE','RG','RI'], index=['CITY_EN'],columns=['DATE'])
WEATHER_TAB.columns = Const_Datestr3('RC_', '2015-06-26','2016-11-20') + Const_Datestr3('RE_', '2015-06-26','2016-11-20') \
+Const_Datestr3('RG_', '2015-06-26','2016-11-20')+Const_Datestr3('RI_', '2015-06-26','2016-11-20')

WEARC_TRN_C = [(lambda x:('RC'+ str(x).zfill(2))) (x)  for x in range(TRN_N)]
WEARE_TRN_C = [(lambda x:('RE'+ str(x).zfill(2))) (x)  for x in range(TRN_N)]
WEARG_TRN_C = [(lambda x:('RG'+ str(x).zfill(2))) (x)  for x in range(TRN_N)]
WEARI_TRN_C = [(lambda x:('RI'+ str(x).zfill(2))) (x)  for x in range(TRN_N)]

WEARD_TST_C = [(lambda x:('RD'+ str(x).zfill(2))) (x)  for x in range(TST_N)]
WEARF_TST_C = [(lambda x:('RF'+ str(x).zfill(2))) (x)  for x in range(TST_N)]
WEARH_TST_C = [(lambda x:('RH'+ str(x).zfill(2))) (x)  for x in range(TST_N)]
WEARJ_TST_C = [(lambda x:('RJ'+ str(x).zfill(2))) (x)  for x in range(TST_N)]


weather_output_columns = (WEARC_TRN_C +  WEARE_TRN_C + WEARG_TRN_C  + WEARI_TRN_C   \
                         + WEARD_TST_C +  WEARF_TST_C + WEARH_TST_C  + WEARJ_TST_C ) 


WEATHER_ALL = pd.DataFrame()
for TRN_STA,TRN_END,TST_STA,TST_END in date_all: 
    weather_input_columns = (Const_Datestr3('RC_', TRN_STA,TRN_END) + Const_Datestr3('RE_', TRN_STA,TRN_END) \
+ Const_Datestr3('RG_', TRN_STA,TRN_END) + Const_Datestr3('RI_', TRN_STA,TRN_END)   \
+ Const_Datestr3('RC_', TST_STA,TST_END) + Const_Datestr3('RE_', TST_STA,TST_END) \
+ Const_Datestr3('RG_', TST_STA,TST_END)+Const_Datestr3('RI_', TST_STA,TST_END)  )


    tt  = WEATHER_TAB[ weather_input_columns]  
    tt.columns = weather_output_columns
    tt.reset_index(level=0, inplace=True)   
    tt.loc[:,'TRN_STA'] = TRN_STA
    tt.loc[:,'TRN_END'] = TRN_END     
    tt.loc[:,'TST_STA'] = TST_STA
    tt.loc[:,'TST_END'] = TST_END   
    
    WEATHER_ALL = pd.concat([WEATHER_ALL,tt],axis = 0)


TRAIN_OK = pd.merge(TRAIN_OK ,WEATHER_ALL,on=['CITY_EN','TRN_STA','TRN_END','TST_STA','TST_END'],how = 'left')
TEST = pd.merge(TEST ,WEATHER_ALL,on=['CITY_EN','TRN_STA','TRN_END','TST_STA','TST_END'],how = 'left')



#%%  readin regional data
#
ALL_FEATURE_LIST = (TRAIN_TRN_C + SHOP_columns + HOLI_TRN_CA + HOLI_TST_CA + HOLI_RATIO + PRECIP_TRN_C +PRECIP_TST_C + weather_output_columns )

X = TRAIN_OK[ALL_FEATURE_LIST]
Y = TRAIN_OK[TRAIN_TST_C]
X_test = TEST[ALL_FEATURE_LIST]

#Xout1 = X[Y[Y<0.25].sum(axis = 1) < 1]
#Yout1 = Y[Y[Y<0.25].sum(axis = 1) < 1]
#
#Xout2 = Xout1[Yout1[Yout1>1.6].sum(axis = 1) < 1]
#Yout2 = Yout1[Yout1[Yout1>1.6].sum(axis = 1) < 1]

#Xout2.to_csv('../XY/0226_X_clean.csv', index = False)
#Yout2.to_csv('../XY/0226_Y_clean.csv', index = False)
#X_test.to_csv('../XY/0226_Xtest_clean.csv', index = False)

X.to_csv('X.csv', index = False)
Y.to_csv('Y.csv', index = False)
X_test.to_csv('Xtest.csv', index = False)
