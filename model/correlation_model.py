# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:36:22 2017

@author: Flamingo
"""

## model of correlation coefficient


import pandas as pd
import numpy as np
import datetime
import copy

import sys
sys.path.append('../TOOLS')
from IJCAI2017_TOOL import *


TEST = pd.read_csv('../feature/TEST_SELLS.csv')

Cor_array = []
for ind,value in TEST.iterrows():
    tt2 = value[np.arange(1,22)].transpose()
    tt3 = np.asarray(tt2,dtype = np.float32).reshape([7,3],order = 'F')
    tt4 = pd.DataFrame( tt3 )
    tt5 = np.corrcoef(tt3,rowvar=0)
    tt6 = tt5[np.triu_indices(3,k=1)]
    Cor_array.append(tt6)

Cor_columns = [(lambda x:('Cor'+ str(x).zfill(2))) (x)  for x in range( 3 )]
Cor_columns_in = ['Cor00','Cor01','Cor04']
Cor_array_pd =  pd.DataFrame(Cor_array, columns = Cor_columns_in)

TEST = pd.concat([TEST,Cor_array_pd], axis=1)

TEST['Cor_mean'] = TEST[['Cor00','Cor01','Cor04']].mean(axis = 1)

TEST_cor = TEST.sort_values(by = ['Cor_mean','Cor04'], ascending  = False )

## process

HIGH_LIMIT = 0.7
CRITERIA = 0.75

TEST_cor_candidate = TEST_cor[TEST_cor['Cor_mean']>0.75]
TEST_cor_candidate['Cor_min'] = TEST_cor_candidate[['Cor_mean','Cor04']].min(axis = 1)
TEST_cor_candidate['credit'] = (TEST_cor_candidate['Cor_min']-CRITERIA)*HIGH_LIMIT/ (1.0- CRITERIA )
TEST_cor_candidate = TEST_cor_candidate[TEST_cor_candidate['credit']>0].sort_values(by = ['credit'], ascending  = False)
TEST_cor_candidate.index = np.arange( len(TEST_cor_candidate) )

average_list = []
increase_list1 = []
increase_list2 = []
for ind,value in TEST_cor_candidate.iterrows():
    w1 = value[np.arange(1,8)].values
    w2 = value[np.arange(8,15)].values
    w3 = value[np.arange(15,22)].values
    w_average = (w1 +w2 +w3)/3.0
    average_list.append(w_average)
    increase_list1.append((0.62*(w3.mean() - w2.mean()) + 0.38*(w2.mean() - w1.mean())))
    increase_list2.append((0.62*(np.median(w3) - np.median(w2)) + 0.38*(np.median(w2)- np.median(w1))))
    

average_list_pd = pd.DataFrame(average_list,columns = np.arange(7))
increase_list1_pd = pd.DataFrame(increase_list1,columns = ['increase1'])
increase_list2_pd = pd.DataFrame(increase_list2,columns = ['increase2'])

TEST_cor_candidate = pd.concat([TEST_cor_candidate,average_list_pd,increase_list1_pd,increase_list2_pd ], axis = 1)
TEST_cor_candidate['7mean'] = TEST[np.arange(1,8)].mean(axis = 1)
TEST_cor_candidate = TEST_cor_candidate[np.abs(TEST_cor_candidate['increase1']/TEST_cor_candidate['7mean']) < 0.25]
TEST_cor_candidate = TEST_cor_candidate[np.abs(TEST_cor_candidate['increase2']/TEST_cor_candidate['7mean']) < 0.25]
TEST_cor_candidate['increase'] = (0.5* TEST_cor_candidate['increase2'] + 0.5 * TEST_cor_candidate['increase1'])*1.0

TEST_cor_candidate[['01','02','03','04','05','06','07' ]]  = TEST_cor_candidate[ np.arange(7) ].sub(-0.9*TEST_cor_candidate['increase'],axis = 0)
TEST_cor_candidate[[ '08','09','10','11','12','13','14'  ]]  = TEST_cor_candidate[ np.arange(7) ].sub(-1.7*TEST_cor_candidate['increase'],axis = 0)


colo_list = ['SHOP_ID','01','02','03','04','05','06','07', '08','09','10','11','12','13','14' ,'credit']
TEST_cor_candidate = TEST_cor_candidate[colo_list]


TEST_cor_candidate.to_csv('cor_model.csv',index = False)
