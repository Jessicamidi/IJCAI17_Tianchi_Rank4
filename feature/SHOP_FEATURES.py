# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 12:48:37 2017

@author: Flamingo
"""

import copy
import sys
sys.path.append('../TOOLS')
from IJCAI2017_TOOL import *
    
SHOP_INFO_EN = pd.read_csv('../data_new/SHOP_INFO_EN.csv')

#%% category infomation

category = pd.read_csv('../data_new/SHOP_CAT.csv')
one_hot = pd.get_dummies(category['CAT'])
category = category.join(one_hot)
SHOP_SJ = map(lambda x:'SJ'+ str(x).zfill(2), np.arange(15)) 
category.columns = ['SHOP_CA1_EN','SHOP_CA2_EN','SHOP_CA3_EN','Num', 'CAT'] + SHOP_SJ
SHOP_INFO_EN = pd.merge(SHOP_INFO_EN, category, on = ['SHOP_CA1_EN','SHOP_CA2_EN','SHOP_CA3_EN'],how = 'left')

#%%


SHOP_LOC_N = SHOP_INFO_EN.groupby('SHOP_LOC',as_index=False).count().loc[:,['SHOP_LOC','SHOP_ID']].rename(columns = {'SHOP_ID':'SHOP_LOC_N'})
SHOP_INFO_EN = pd.merge(SHOP_INFO_EN,SHOP_LOC_N, on=['SHOP_LOC'], how= 'left')


SHOP_INFO_EN['SHOP_SCO'] = SHOP_INFO_EN['SHOP_SCO'].fillna(SHOP_INFO_EN['SHOP_SCO'].mean() )
SHOP_INFO_EN['SHOP_COM'] = SHOP_INFO_EN['SHOP_COM'].fillna(SHOP_INFO_EN['SHOP_COM'].mean() )
SHOP_INFO_EN = SHOP_INFO_EN[['SHOP_ID','CITY_EN','SHOP_PAY','SHOP_SCO','SHOP_COM','SHOP_LEV','SHOP_LOC_N'] + SHOP_SJ]
SHOP_SD = map(lambda x:'SD'+ str(x).zfill(2), np.arange(5)) 
SHOP_INFO_EN.columns = ['SHOP_ID','CITY_EN'] + SHOP_SD + SHOP_SJ

PAYNW = pd.read_csv('../data_new/user_pay_new.csv')
VIENW = pd.read_csv('../data_new/user_view_new.csv')
#%%
HOLI = pd.read_csv('../additional/HOLI.csv')
HOLI['DATE'] = [ (lambda x: str(datetime.datetime.strptime(str(x),'%Y%m%d').date() ) )(x) for x in HOLI['DATE'] ] 

#%%  calculate top hours   SE,SF

TOP_N = 1
SHOP_ID = []
SHOP_HOUR_head = []
SHOP_PCT_head = []
for SHOP_IND in range(1,2001):
    tt = PAYNW[PAYNW['SHOP_ID'] == SHOP_IND]
    tt2 = tt.groupby('HOUR',as_index = False).sum()
    tt3 = tt2.sort_values('Num_post',ascending = False,inplace = False)
    tt4 = tt3.head(TOP_N)['HOUR'].values
    tt5 = tt3.head(TOP_N)['Num_post'].values/tt3['Num_post'].sum()
    SHOP_ID.append(SHOP_IND)
    SHOP_HOUR_head.append(tt4)
    SHOP_PCT_head.append(tt5)
SHOP_ID_df = pd.DataFrame(SHOP_ID)
SHOP_HOUR_head_df = pd.DataFrame(SHOP_HOUR_head)
SHOP_PCT_head_df = pd.DataFrame(SHOP_PCT_head)

SELL_INFO = pd.concat([SHOP_ID_df,SHOP_HOUR_head_df,SHOP_PCT_head_df],axis = 1)
SHOP_SE =  [(lambda x:('SE'+ str(x).zfill(2))) (x)  for x in range(TOP_N)]
SHOP_SF =  [(lambda x:('SF'+ str(x).zfill(2))) (x)  for x in range(TOP_N)]
SELL_INFO.columns = ['SHOP_ID'] + SHOP_SE + SHOP_SF
 
#%%  calculate top hours 
SHOP_ID = []
SHOP_OPEN = []
SHOP_CLOSE = []
SHOP_LAST = []
SHOP_MEAN = []
for SHOP_IND in range(1,2001):
    tt = PAYNW[PAYNW['SHOP_ID'] == SHOP_IND]
    tt2 = tt.groupby('DATE',as_index = False).min().mean()
    tt3 = tt.groupby('DATE',as_index = False).max().mean()
    tt['MEAN']  = tt['Num_post'] * tt['HOUR'] 
    SHOP_ID.append(SHOP_IND)    
    SHOP_OPEN.append(tt2.HOUR)
    SHOP_CLOSE.append(tt3.HOUR)
    SHOP_LAST.append(tt3.HOUR -tt2.HOUR   )  
    SHOP_MEAN.append(tt['MEAN'].sum()/tt['Num_post'].sum() )
SHOP_ID_df = pd.DataFrame(SHOP_ID)
SHOP_OPEN_df = pd.DataFrame(SHOP_OPEN )
SHOP_CLOSE_df = pd.DataFrame(SHOP_CLOSE )   
SHOP_LAST_df = pd.DataFrame(SHOP_LAST )
SHOP_MEAN_df = pd.DataFrame(SHOP_MEAN )
HOUR_INFO = pd.concat([SHOP_ID_df,SHOP_OPEN_df,SHOP_CLOSE_df, SHOP_LAST_df,SHOP_MEAN_df],axis = 1)
SHOP_SG = map(lambda x:'SG'+ str(x).zfill(2), np.arange(4))
HOUR_INFO.columns = ['SHOP_ID'] + SHOP_SG 

#%%  need to fillna 0
PAYNW_gp = PAYNW.groupby(['DATE','SHOP_ID'],as_index = False).sum()
VIENW_gp = VIENW.groupby(['DATE','SHOP_ID'],as_index = False).sum()
PAYNW_gp['DofW'] = Datestr2DofW(PAYNW_gp['DATE'] )
VIENW_gp['DofW'] = Datestr2DofW(VIENW_gp['DATE'] )
PAYNW_gp = pd.merge(PAYNW_gp,HOLI,on = ['DATE'],how = 'left')
VIENW_gp = pd.merge(VIENW_gp,HOLI,on = ['DATE'],how = 'left')

PAYNW_VIENW = pd.merge(PAYNW_gp,VIENW_gp, on =['DATE','SHOP_ID'], how = 'inner' )
PAYNW_VIENW['RATIO'] = PAYNW_VIENW['Num_post_y'] / PAYNW_VIENW['Num_post_x']
SHOP_PAYNW_VIENW = PAYNW_VIENW.groupby('SHOP_ID',as_index = False).mean()
SHOP_SC = 'SC00'    ### view/pay ratio
RATIO_INFO = SHOP_PAYNW_VIENW[['SHOP_ID','RATIO']].rename(columns = {'RATIO':SHOP_SC})


#%%  first online date,  and online days to count
GAP_INFO = PAYNW.groupby('SHOP_ID',as_index = False).min().loc[:,['SHOP_ID','DATE']]
GAP_INFO['RAT_DAY'] = pd.to_datetime(  GAP_INFO['DATE']) - datetime.date(2015,6,26)
GAP_INFO['RAT_DAY'] = [(lambda x:(x.days)) (x) for x in GAP_INFO['RAT_DAY'] ]
SHOP_SH =  [(lambda x:('SH'+ str(x).zfill(2))) (x)  for x in range(2)]
GAP_INFO.columns = ['SHOP_ID'] + SHOP_SH

#%%  ratio of weekday and weekend median


PAYNW_gp_wd = PAYNW_gp[PAYNW_gp['HOLI']==0].groupby('SHOP_ID',as_index = False).median()
PAYNW_gp_wd = PAYNW_gp_wd[['SHOP_ID','Num_post']].rename(columns = {'Num_post':'WD_MED'})
PAYNW_gp_wk = PAYNW_gp[PAYNW_gp['HOLI']>0].groupby('SHOP_ID',as_index = False).median()
PAYNW_gp_wk = PAYNW_gp_wk[['SHOP_ID','Num_post']].rename(columns = {'Num_post':'WK_MED'})
PAYNW_gp_wdwk = pd.merge(PAYNW_gp_wd, PAYNW_gp_wk, on='SHOP_ID',how = 'left')

#%%
HOLI_list = [0,0,0,0,0,1,1]
DofW_list = [0,1,2,3,4,5,6]

for holi_ind, DofW_ind in zip(HOLI_list,DofW_list):
    DAYOFWEEk = PAYNW_gp[(PAYNW_gp['HOLI']==holi_ind)&(PAYNW_gp['DofW']==DofW_ind)].groupby('SHOP_ID',as_index = False).median()
    DAYOFWEEk = DAYOFWEEk[['SHOP_ID','Num_post']].rename(columns = {'Num_post':'d'+str(DofW_ind)})
    PAYNW_gp_wdwk = pd.merge(PAYNW_gp_wdwk, DAYOFWEEk, on='SHOP_ID',how = 'left')

SHOP_SI =  [(lambda x:('SI'+ str(x).zfill(2))) (x)  for x in range(9)]
PAYNW_gp_wdwk.columns = ['SHOP_ID'] + SHOP_SI
PAYNW_gp_wdwk['FIX'] = PAYNW_gp_wdwk[SHOP_SI].mean(axis = 1)
PAYNW_gp_wdwk[SHOP_SI] = PAYNW_gp_wdwk[SHOP_SI].div( PAYNW_gp_wdwk['FIX'],axis = 0)
del PAYNW_gp_wdwk['FIX']
PAYNW_gp_wdwk['ratio'] = PAYNW_gp_wdwk['SI00']/ PAYNW_gp_wdwk['SI01']
SHOP_SI =  [(lambda x:('SI'+ str(x).zfill(2))) (x)  for x in range(10)]
PAYNW_gp_wdwk.columns = ['SHOP_ID'] + SHOP_SI


#%%
SHOP_INFO_EN = pd.merge(SHOP_INFO_EN,SELL_INFO, on='SHOP_ID',how = 'left')   #SA
SHOP_INFO_EN = pd.merge(SHOP_INFO_EN,HOUR_INFO, on='SHOP_ID',how = 'left')  # SE,SF
SHOP_INFO_EN = pd.merge(SHOP_INFO_EN,RATIO_INFO, on='SHOP_ID',how = 'left')  #SC
SHOP_INFO_EN = pd.merge(SHOP_INFO_EN,GAP_INFO, on='SHOP_ID',how = 'left')    #SH
SHOP_INFO_EN = pd.merge(SHOP_INFO_EN,PAYNW_gp_wdwk, on='SHOP_ID',how = 'left')    #SI
SHOP_INFO_EN = SHOP_INFO_EN.fillna(0)


#%%
SHOP_INFO_EN.to_csv('SHOP_FEATURES.csv',index = False)

