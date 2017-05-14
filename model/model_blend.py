# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 16:32:52 2017

@author: Flamingo
"""
#%%

import sys
sys.path.append('../TOOLS')
from IJCAI2017_TOOL import *
Day_list =  [(lambda x:('D'+ str(x).zfill(2))) (x)  for x in range(1,15)]
result_columns =['SHOP_ID'] + Day_list
result0 = pd.read_csv('benchmark.csv',names = result_columns)

result_merge1 = pd.read_csv('xgb_model1.csv',names = result_columns)
result_merge2 = pd.read_csv('GBDT_model.csv',names = result_columns)
result_merge3 = pd.read_csv('xgb_model2.csv',names = result_columns)

result_merged  = result_merge1.copy()
result_merged[Day_list] = 0.47* result_merge1[Day_list] + 0.19*result_merge2[Day_list]  + 0.34*result_merge3[Day_list]
result_merged['SHOP_ID'] = result_merged['SHOP_ID'].astype(int)


result = result_merged.copy()




print( len( result_merge1.groupby(['SHOP_ID'],as_index = False).count()) )
print( len( result_merge2.groupby(['SHOP_ID'],as_index = False).count()) )
print( len( result_merge3.groupby(['SHOP_ID'],as_index = False).count()) )
print( len( result_merged.groupby(['SHOP_ID'],as_index = False).count()) )


SHOP_INFO_EN = pd.read_csv('../data_new/SHOP_INFO_EN.csv')

file_ratio1 = pd.DataFrame(result_merge1 /result0)
file_ratio2 = pd.DataFrame(result_merge2 /result0)
file_ratio3 = pd.DataFrame(result_merge3 /result0)
file_ratio_merge = pd.DataFrame(result_merged /result0)
print(file_ratio1.mean(axis=0).mean())
print(file_ratio2.mean(axis=0).mean())
print(file_ratio3.mean(axis=0).mean())
print(file_ratio_merge.mean(axis=0).mean())

fig, ax = plt.subplots()
fig.patch.set_facecolor('white')

font = {'family': 'serif',
        'color':  'k',
        'weight': 'normal',
        'size': 20,
        }
        
plt.xlabel('ratio', fontdict=font)


#%%  add cor model 

Cor_model = pd.read_csv('Cor_model.csv')
Day_list2 =  [(lambda x:('Cor_D'+ str(x).zfill(2))) (x)  for x in range(1,15)]
Cor_model.columns = ['SHOP_ID'] + Day_list2 + ['credit']


Cor_model['SHOP_ID'] = Cor_model['SHOP_ID'].astype(int)


result = pd.merge(result,Cor_model,on = ['SHOP_ID'],how = 'left' )

print('step5:number of shops are:')
print( len( result.groupby(['SHOP_ID'],as_index = False).count()) )

null_ind = result['credit'].isnull()
tt1 = result.loc[~null_ind, Day_list].multiply(1.0-result.loc[~null_ind, 'credit'],axis = 0)
tt2 = result.loc[~null_ind, Day_list2].multiply(result.loc[~null_ind, 'credit'],axis = 0)
tt2.columns = Day_list
tt3 = tt1 + tt2

for ind,value in tt3.iterrows():
    result.loc[ind,Day_list] = value.values
result = result[result_columns]


file_ratio_merge1 = pd.DataFrame(result /result0)
plt.hist(file_ratio_merge1[Day_list].values.reshape(-1),bins = np.linspace(0.8,1.3,101),alpha = 0.5)
#%%


DOU11 = pd.read_csv('DOU11_coef.csv')

result = pd.merge(result, SHOP_INFO_EN, on=['SHOP_ID'],how = 'left')
result = pd.merge(result, DOU11, on=['SHOP_ID'],how = 'left')


result['D11'] = result['D11'] * (result['DOU11']*0.8 + 1.0*(0.2) )
result['D12'] = result['D12'] * (result['DOU11']*0.2+ 1.0*0.8)
result['D13'] = result['D13'] * (result['DOU11']*0.1+ 1.0*0.9)

result = result[result_columns]


file_ratio_merge1 = pd.DataFrame(result /result0)
plt.hist(file_ratio_merge1[Day_list].values.reshape(-1),bins = np.linspace(0.8,1.3,101),alpha = 0.5,color = 'red')


print('number of shops are:')
print( len( result.groupby(['SHOP_ID'],as_index = False).count()) )


#%%
result = result.round()
result = result.astype(np.int)
filename = 'FINAL_RESULT.csv'
result.to_csv(filename,header=None, index = False)