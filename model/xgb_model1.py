# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 01:06:37 2017

@author: Flamingo
"""

import sys
sys.path.append('../TOOLS')
from IJCAI2017_TOOL import *

#%%   readin data

X = pd.read_csv('../feature/X.csv')
Y = pd.read_csv('../feature/Y.csv')
Xtest = pd.read_csv('../feature/Xtest.csv',low_memory=False)
fix = pd.read_csv('../feature/FIXER.csv')

#%%  preprocess
TRN_N = 21
TST_N = 14
TST_PAD_N = 14 + 4

TRN_RANGE = np.arange(0,TRN_N)
TST_RANGE = np.arange(0,TST_N)

TRAIN_TRN_C = [(lambda x:('SA'+ str(x).zfill(2))) (x)  for x in TRN_RANGE]
TRAIN_TST_C = [(lambda x:('SB'+ str(x).zfill(2))) (x)  for x in TST_RANGE] 

SHOP_SC = ['SC00']
SHOP_SD = [(lambda x:('SD'+ str(x).zfill(2))) (x)  for x in range(5)]
SHOP_SE =  [(lambda x:('SE'+ str(x).zfill(2))) (x)  for x in range(1)]
SHOP_SF = [(lambda x:('SF'+ str(x).zfill(2))) (x)  for x in range(1)]
SHOP_SG = [(lambda x:('SG'+ str(x).zfill(2))) (x)  for x in range(4)]
SHOP_SH = [(lambda x:('SH'+ str(x).zfill(2))) (x)  for x in range(1,2)]
SHOP_SI = [(lambda x:('SI'+ str(x).zfill(2))) (x)  for x in range(10)]
SHOP_SJ = [(lambda x:('SJ'+ str(x).zfill(2))) (x)  for x in range(15)]
SHOP_columns = SHOP_SC +SHOP_SD+SHOP_SE+SHOP_SF+SHOP_SG + SHOP_SH  + SHOP_SI + SHOP_SJ


WEARC_TRN_C = [(lambda x:('RC'+ str(x).zfill(2))) (x)  for x in range(TRN_N)]
WEARD_TST_C = [(lambda x:('RD'+ str(x).zfill(2))) (x)  for x in range(TST_N)]
WEARE_TRN_C = [(lambda x:('RE'+ str(x).zfill(2))) (x)  for x in range(TRN_N)]
WEARF_TST_C = [(lambda x:('RF'+ str(x).zfill(2))) (x)  for x in range(TST_N)]
WEARG_TRN_C = [(lambda x:('RG'+ str(x).zfill(2))) (x)  for x in range(TRN_N)]
WEARH_TST_C = [(lambda x:('RH'+ str(x).zfill(2))) (x)  for x in range(TST_N)]
WEARI_TRN_C = [(lambda x:('RI'+ str(x).zfill(2))) (x)  for x in range(TRN_N)]
WEARJ_TST_C = [(lambda x:('RJ'+ str(x).zfill(2))) (x)  for x in range(TST_N)]

weather_output_columns = WEARC_TRN_C + WEARD_TST_C + WEARE_TRN_C + WEARF_TST_C +WEARG_TRN_C + WEARH_TST_C  + WEARI_TRN_C + WEARJ_TST_C 
#weather_output_columns = WEARE_TRN_C + WEARF_TST_C +WEARG_TRN_C + WEARH_TST_C  + WEARI_TRN_C + WEARJ_TST_C 
HOLI_TRN_CA  = [(lambda x:('NC'+ str(x).zfill(2))) (x)  for x in TRN_RANGE]
HOLI_TST_CA  = [(lambda x:('ND'+ str(x).zfill(2))) (x)  for x in range(-2,TST_PAD_N-2)]
HOLI_RATIO = [(lambda x:('NE'+ str(x).zfill(2))) (x)  for x in range(1)]
PRECIP_TRN_C = [(lambda x:('RA'+ str(x).zfill(2))) (x)  for x in TRN_RANGE]
PRECIP_TST_C = [(lambda x:('RB'+ str(x).zfill(2))) (x)  for x in range(TST_N)]

X[TRAIN_TRN_C] = X[TRAIN_TRN_C].T.fillna(X[TRAIN_TRN_C].mean(axis = 1)).T
Xtest[TRAIN_TRN_C] = Xtest[TRAIN_TRN_C].T.fillna(Xtest[TRAIN_TRN_C].mean(axis = 1)).T

TRN_BASE = X[TRAIN_TRN_C].median(axis=1)
TST_BASE = Xtest[TRAIN_TRN_C].median(axis = 1)

X[TRAIN_TRN_C] = X[TRAIN_TRN_C].sub(TRN_BASE , axis =0)
Xtest[TRAIN_TRN_C] = Xtest[TRAIN_TRN_C].sub(TST_BASE , axis =0)
Y[TRAIN_TST_C] = Y[TRAIN_TST_C].sub(TRN_BASE , axis =0)

COLUMN_ALL = TRAIN_TRN_C + HOLI_TRN_CA +  HOLI_TST_CA  + HOLI_RATIO + SHOP_columns + PRECIP_TRN_C  + WEARC_TRN_C  + WEARE_TRN_C + WEARG_TRN_C + WEARI_TRN_C 

#%%
#np.random.seed(0)
#ran_ind  = np.random.randint(X.shape[0],size=200)
#X = X.loc[ran_ind,:]
#Y = Y.loc[ran_ind,:]

def abs_relative_error(y_pred,y_true):
    return np.mean(np.mean(np.abs(y_pred-y_true)/np.abs(y_pred+y_true)) )
def abs_relative_error_element(y_pred,y_true):
    return np.abs(y_pred-y_true)/np.abs(y_pred+y_true)
    
def abs_error(y_pred,y_true):
    return np.mean(np.mean(np.abs(y_pred-y_true)))
    
def abs_error_element(y_pred,y_true):
    return np.abs(y_pred-y_true)  


import xgboost as xgb

XGBR = xgb.XGBRegressor(max_depth = 3,learning_rate=0.1,n_estimators=500)


Ytrain_all = pd.DataFrame()
error_list = []
for ind in range(14):
    if ind<= 1:
        COLUMN_IND1 = [(lambda x:('RB'+ str(x).zfill(2))) (x)  for x in range(0 ,ind+2 )]
        COLUMN_IND2 = [(lambda x:('RF'+ str(x).zfill(2))) (x)  for x in range(0 ,ind+2 )]
        COLUMN_IND3 = [(lambda x:('RH'+ str(x).zfill(2))) (x)  for x in range(0 ,ind+2 )]
        COLUMN_IND4  = [(lambda x:('RJ'+ str(x).zfill(2))) (x)  for x in range(0 ,ind+2 )]
        COLUMN_IND5 = [(lambda x:('RD'+ str(x).zfill(2))) (x)  for x in range(0 ,ind+2 )]
    elif(ind<= 12):
        COLUMN_IND1 = [(lambda x:('RB'+ str(x).zfill(2))) (x)  for x in range(ind-2 ,ind+2 )]
        COLUMN_IND2 = [(lambda x:('RF'+ str(x).zfill(2))) (x)  for x in range(ind-2 ,ind+2 )]
        COLUMN_IND3 = [(lambda x:('RH'+ str(x).zfill(2))) (x)  for x in range(ind-2 ,ind+2 )]
        COLUMN_IND4  = [(lambda x:('RJ'+ str(x).zfill(2))) (x)  for x in range(ind-2 ,ind+2 )]
        COLUMN_IND5 = [(lambda x:('RD'+ str(x).zfill(2))) (x)  for x in range(ind-2 ,ind+2 )]
    else:
        COLUMN_IND1 = [(lambda x:('RB'+ str(x).zfill(2))) (x)  for x in range(ind-2 ,ind+1 )]
        COLUMN_IND2 = [(lambda x:('RF'+ str(x).zfill(2))) (x)  for x in range(ind-2 ,ind+1 )]
        COLUMN_IND3 = [(lambda x:('RH'+ str(x).zfill(2))) (x)  for x in range(ind-2 ,ind+1 )]
        COLUMN_IND4  = [(lambda x:('RJ'+ str(x).zfill(2))) (x)  for x in range(ind-2 ,ind+1 )] 
        COLUMN_IND5 = [(lambda x:('RD'+ str(x).zfill(2))) (x)  for x in range(ind-2 ,ind+1 )]
    
    COLUMN_IND = COLUMN_IND1 +  COLUMN_IND2 +COLUMN_IND3 +  COLUMN_IND4  + COLUMN_IND5

    Xtrain_in = X[COLUMN_ALL + COLUMN_IND]
    time1 = time.time()
    XGBR.fit(Xtrain_in.values,Y.values[:,ind] )
    y_true = Y.values[:,ind]
    y_pred = XGBR.predict(Xtrain_in.values)
    time2 = time.time()    
    print(str(ind)+ '_error:' + str(abs_error(y_pred,y_true  ) ) + '__time:'+ str(time2 - time1) )
    error_list.append(abs_error(y_pred, y_true  ))
    
    Ytrain =  XGBR.predict(Xtrain_in.values)
    Ytrain_df = pd.DataFrame(Ytrain)
    Ytrain_all = pd.concat((Ytrain_all,Ytrain_df),axis = 1)   
print(np.mean(error_list))

#%%

Ytrain_true = pd.DataFrame(Y)
Y_error = abs_error_element(Ytrain_all.values,Ytrain_true.values  )  
Good_ind =  Y_error.sum(axis=1).argsort()[0:np.int(0.90*len(X))]
#%%

XGBR = xgb.XGBRegressor(max_depth = 5,learning_rate=0.03,n_estimators=1600,reg_alpha=1,reg_lambda=0)
Ytest_all = pd.DataFrame()
error_list = []
for ind in range(14):
    
    if ind<= 1:
        COLUMN_IND1 = [(lambda x:('RB'+ str(x).zfill(2))) (x)  for x in range(0 ,ind+2 )]
        COLUMN_IND2 = [(lambda x:('RF'+ str(x).zfill(2))) (x)  for x in range(0 ,ind+2 )]
        COLUMN_IND3 = [(lambda x:('RH'+ str(x).zfill(2))) (x)  for x in range(0 ,ind+2 )]
        COLUMN_IND4  = [(lambda x:('RJ'+ str(x).zfill(2))) (x)  for x in range(0 ,ind+2 )]
        COLUMN_IND5 = [(lambda x:('RD'+ str(x).zfill(2))) (x)  for x in range(0 ,ind+2 )]
    elif(ind<= 12):
        COLUMN_IND1 = [(lambda x:('RB'+ str(x).zfill(2))) (x)  for x in range(ind-2 ,ind+2 )]
        COLUMN_IND2 = [(lambda x:('RF'+ str(x).zfill(2))) (x)  for x in range(ind-2 ,ind+2 )]
        COLUMN_IND3 = [(lambda x:('RH'+ str(x).zfill(2))) (x)  for x in range(ind-2 ,ind+2 )]
        COLUMN_IND4  = [(lambda x:('RJ'+ str(x).zfill(2))) (x)  for x in range(ind-2 ,ind+2 )]
        COLUMN_IND5 = [(lambda x:('RD'+ str(x).zfill(2))) (x)  for x in range(ind-2 ,ind+2 )]

    else:
        COLUMN_IND1 = [(lambda x:('RB'+ str(x).zfill(2))) (x)  for x in range(ind-2 ,ind+1 )]
        COLUMN_IND2 = [(lambda x:('RF'+ str(x).zfill(2))) (x)  for x in range(ind-2 ,ind+1 )]
        COLUMN_IND3 = [(lambda x:('RH'+ str(x).zfill(2))) (x)  for x in range(ind-2 ,ind+1 )]
        COLUMN_IND4  = [(lambda x:('RJ'+ str(x).zfill(2))) (x)  for x in range(ind-2 ,ind+1 )] 
        COLUMN_IND5 = [(lambda x:('RD'+ str(x).zfill(2))) (x)  for x in range(ind-2 ,ind+1 )]


    COLUMN_IND = COLUMN_IND1 +  COLUMN_IND2 +COLUMN_IND3 +  COLUMN_IND4  + COLUMN_IND5
    Xtrain_in = X[COLUMN_ALL + COLUMN_IND]
    Xtest_in = Xtest[COLUMN_ALL + COLUMN_IND]
    time1 = time.time()
    XGBR.fit(Xtrain_in.values[Good_ind],Y.values[Good_ind,ind] )
    y_true = Y.values[Good_ind,ind]
    y_pred = XGBR.predict(Xtrain_in.values[Good_ind])
    time2 = time.time()    
    print(str(ind)+ '_error:' + str(abs_error(y_pred,y_true  ) ) + '__time:'+ str(time2 - time1) )
    error_list.append(abs_error(y_pred, y_true  ))
    Ytest = XGBR.predict(Xtest_in.values)
    Ytest_df = pd.DataFrame(Ytest)
    Ytest_all = pd.concat((Ytest_all,Ytest_df),axis = 1)   
print(np.mean(error_list))

##%%   fix 
result = Ytest_all.copy()
result.columns = [np.arange(14)]
result['SHOP_ID'] = np.arange(1,2001)
result = pd.merge(result ,fix,on='SHOP_ID',how = 'left')
result[np.arange(0,14)] = np.e**(result[np.arange(0,14)].sub(-TST_BASE, axis = 0).multiply(np.log(result['VALUE']),axis=0) )

#%%   submit 
result_sub = pd.concat((result['SHOP_ID'],result[np.arange(0,14)] ),axis = 1 )
result_sub[result_sub<0] = 5


filename = 'xgb_model1.csv'
result_sub.to_csv(filename,header=None, index = False)
#%% 
