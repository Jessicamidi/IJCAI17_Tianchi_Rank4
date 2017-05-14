# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 22:07:17 2017

@author: Flamingo
"""

#%%
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime

user_cols = ['USER_ID','SHOP_ID','TIME_STA']
user_pay = pd.read_csv('../data/user_pay.txt',header=None, names = user_cols)
user_view = pd.read_csv('../data/user_view.txt',header=None,names = user_cols)
user_view_extra = pd.read_csv('../data/extra_user_view.txt',header=None,names = user_cols)
user_view = pd.concat([user_view, user_view_extra])

#%%

user_pay['TIME_STA'] = pd.to_datetime(user_pay['TIME_STA'])
user_pay['DATE'] = pd.to_datetime(user_pay['TIME_STA']).dt.date
user_pay['HOUR'] = pd.to_datetime(user_pay['TIME_STA']).dt.hour

user_pay_new = user_pay.groupby(by =['USER_ID','SHOP_ID','DATE','HOUR'],as_index = False).count()
user_pay_new = user_pay_new.rename(columns={'TIME_STA':'Num_raw'})
user_pay_new['Num_post'] = 1+ np.log(user_pay_new['Num_raw'])/ np.log(2)
user_pay_new = user_pay_new.groupby(by =['SHOP_ID','DATE','HOUR'],as_index = False).sum()
user_pay_new['DofW'] = pd.to_datetime(user_pay_new['DATE']).dt.dayofweek
user_pay_new = user_pay_new.drop('USER_ID', 1)
user_pay_new.to_csv('user_pay_new.csv',index = False)

#%%
user_view['TIME_STA'] = pd.to_datetime(user_view['TIME_STA'])
user_view['DATE'] = pd.to_datetime(user_view['TIME_STA']).dt.date
user_view['HOUR'] = pd.to_datetime(user_view['TIME_STA']).dt.hour

user_view_new = user_view.groupby(by =['USER_ID','SHOP_ID','DATE','HOUR'],as_index = False).count()
user_view_new = user_view_new.rename(columns={'TIME_STA':'Num_raw'})
user_view_new['Num_post'] = 1+ np.log(user_view_new['Num_raw'])/ np.log(2)
user_view_new = user_view_new.groupby(by =['SHOP_ID','DATE','HOUR'],as_index = False).sum()
user_view_new['DofW'] = pd.to_datetime(user_view_new['DATE']).dt.dayofweek
user_view_new = user_view_new.drop('USER_ID', 1)
user_view_new.to_csv('user_view_new.csv',index = False)

#%%
