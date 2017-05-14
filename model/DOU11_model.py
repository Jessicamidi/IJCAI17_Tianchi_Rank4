# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 14:37:16 2017

@author: Flamingo
"""

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

import sys
sys.path.append('../TOOLS')
from IJCAI2017_TOOL import *

category = pd.read_csv('../data_new/SHOP_CAT.csv')
one_hot = pd.get_dummies(category['CAT'])
category = category.join(one_hot)
SHOP_SJ = map(lambda x:'SJ'+ str(x).zfill(2), np.arange(15)) 
category.columns = ['SHOP_CA1_EN','SHOP_CA2_EN','SHOP_CA3_EN','Num', 'CAT'] + SHOP_SJ

SHOP_INFO_EN = pd.read_csv('../data_new/SHOP_INFO_EN.csv')
SHOP_INFO_EN = pd.merge(SHOP_INFO_EN, category, on = ['SHOP_CA1_EN','SHOP_CA2_EN','SHOP_CA3_EN'],how = 'left')


SHOP_INFO_EN['SHOP_CA1_EN'] = [(lambda x:(str(x.strip()) ) ) (x)  for x in SHOP_INFO_EN['SHOP_CA1_EN']]
PAYNW = pd.read_csv('../data_new/user_pay_new.csv')

PAYNW_TAB = pd.pivot_table(PAYNW, values=['Num_post'], index=['SHOP_ID'],columns=['DATE'], aggfunc=np.sum)   
PAYNW_TAB = pd.concat( [PAYNW_TAB[PAYNW_TAB.columns[0:169:1]], pd.DataFrame({'A':[np.nan],},index=np.arange(1,2001)),PAYNW_TAB[PAYNW_TAB.columns[169::1]] ], axis = 1)
PAYNW_TAB.columns = [str((datetime.datetime.strptime('20150626','%Y%m%d') + datetime.timedelta(days=x)).date()) for x in range( PAYNW_TAB.shape[1])] 

# select shop with double 11 sales
tt = PAYNW_TAB.loc[:,'2015-10-28':'2015-11-26'] 
tt = tt[tt.count(axis = 1)>=30]
tt2 = PAYNW_TAB.loc[PAYNW_TAB.index.isin(tt.index), '2015-10-28':'2015-11-25']
tt2.reset_index(level=0, inplace=True)

# calculate relateIve gain
tt2['ratio'] = tt2['2015-11-11']/( 0.15*tt2['2015-10-28'] + 0.35*tt2['2015-11-04'] + 0.35*tt2['2015-11-18'] + 0.15*tt2['2015-11-25'])

tt2 = pd.merge(tt2,SHOP_INFO_EN, on = ['SHOP_ID'], how = 'left' )

#%%  shop_related_features
SHOP_FEATURE = pd.read_csv("../feature/SHOP_FEATURES.csv",low_memory=False)
SHOP_SC = ['SC00']
SHOP_SD = map(lambda x:'SD'+ str(x).zfill(2), np.arange(6)) 
SHOP_SE = map(lambda x:'SE'+ str(x).zfill(2), np.arange(1))
SHOP_SF = map(lambda x:'SF'+ str(x).zfill(2), np.arange(1))
SHOP_SG = map(lambda x:'SG'+ str(x).zfill(2), np.arange(4))
SHOP_SH = map(lambda x:'SH'+ str(x).zfill(2), np.arange(2))
SHOP_SI =  [(lambda x:('SI'+ str(x).zfill(2))) (x)  for x in range(10)]
SHOP_columns = SHOP_SC + SHOP_SD + SHOP_SE + SHOP_SF + SHOP_SG +  SHOP_SH + SHOP_SI

SHOP_FEATURE = SHOP_FEATURE[ ['SHOP_ID']  + SHOP_SC + SHOP_SE + SHOP_SF + SHOP_SG  + SHOP_SI]


tt2 = pd.merge(tt2[['SHOP_ID','SHOP_PAY','SHOP_SCO','SHOP_COM', 'SHOP_LEV','ratio']+SHOP_SJ], SHOP_FEATURE , on = ['SHOP_ID'],how = 'left')
SHOP_INFO_EN = pd.merge(SHOP_INFO_EN[['SHOP_ID','SHOP_PAY','SHOP_SCO','SHOP_COM', 'SHOP_LEV']+SHOP_SJ], SHOP_FEATURE , on = ['SHOP_ID'],how = 'left')

import copy
SHOP_INFO_EN2 = copy.deepcopy(SHOP_INFO_EN)
del tt2['SHOP_ID']
del SHOP_INFO_EN['SHOP_ID']

Y  = tt2[['ratio']].values
del tt2['ratio']
X  = tt2.values
Xtest = SHOP_INFO_EN.values

XGBR = xgb.XGBRegressor(max_depth = 2,learning_rate=0.01,n_estimators=500,reg_alpha=10,gamma = 1)
XGBR.fit(X,Y)

ytest = XGBR.predict(Xtest)
print(ytest.mean())
print(ytest.min())
print(ytest.max())
print(ytest.std())
SHOP_INFO_EN2['DOU11'] = ytest*0.6+1.0*0.4

SHOP_INFO_EN2.to_csv('DOU11_coef.csv',index = False)



fig, ax = plt.subplots()
fig.patch.set_facecolor('white')

font = {'family': 'serif',
        'color':  'k',
        'weight': 'normal',
        'size': 20,
        }
        
plt.xlabel('ratio', fontdict=font)
plt.hist(SHOP_INFO_EN2['DOU11'].values.reshape(-1),bins = np.linspace(0.8,1.3,21),alpha = 0.5)



