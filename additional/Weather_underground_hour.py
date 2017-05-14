# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 01:01:33 2017

@author: Flamingo
"""
#%%
from bs4 import BeautifulSoup
import urllib
import pandas as pd
import numpy as np

CITY_NAME = pd.read_csv('CITY_NAME2.csv')
PORT_NAME = CITY_NAME[['AIRPORT_CODE','PORT']].groupby('AIRPORT_CODE',as_index=False).count()
for ind,value in PORT_NAME[13:14].iterrows():
    print(value['AIRPORT_CODE'])
#%%
    Port = []
    Date =[]
    Time = []
    Temp = []
    Bodytemp = []
    Dew = []
    Humidity =[]
    Pressure = []
    Visibility =[]
    Wind_dir = []
    Wind_speed = []
    Gust_speed =[]
    Event =[]
    Condition = []


    for vYear in range(2015, 2017):
        if vYear == 2015:
            Monthrange = np.arange(5, 13)
        else:
            Monthrange = np.arange(1, 13)
        for vMonth in Monthrange:
            for vDay in range(1, 32):
                if vYear % 4 == 0:
                    if vMonth == 2 and vDay > 29:
                        break
                else:
                    if vMonth == 2 and vDay > 28:
                        break
                if vMonth in [4, 6, 9, 11] and vDay > 30:
                    break
        #%%
                
                theDate = str(vYear) + "/" + str(vMonth) + "/" + str(vDay)
                theDate2 = str(vYear) + "-" + str(vMonth).zfill(2) + "-" + str(vDay).zfill(2)
                print(theDate2)
                theport = value['AIRPORT_CODE']
                 
                
                theurl = "http://www.wunderground.com/history/airport/"+ theport +"/" + theDate + "/DailyHistory.html?MR=1"
                thepage = urllib.urlopen(theurl)
                soup = BeautifulSoup(thepage, "html.parser")
            
            
                soup_detail = soup.find_all('tr')

                Row_Num = len(soup_detail)
                
                print(Row_Num)
                
                col_count = []
                
                for ind in soup_detail:
                   col_count.append(  len(ind.find_all('td') )  )
                col_count  = np.asarray(col_count )
            #    print(col_count[-1])
                if col_count[-1].tolist() in [12,13]:
                    first_row = np.amin(np.where(col_count == col_count[-1]))       
                    last_row = np.amax(np.where(col_count ==col_count[-1])) 
                    Col_Num = col_count[-1]
                
                for row_ind in np.arange(first_row,last_row+1):
                    soup_detail_line = soup_detail[row_ind].find_all('td')

                    if Col_Num == 13:
                        Col_indadd = 1
                        Bodytemp.append(soup_detail_line[2].text.split()[0])
                    else:
                        Col_indadd = 0    
                        Bodytemp.append('-')
                        
                    Port.append(theport)
                    Date.append(theDate2)
                    Time.append( soup_detail_line[0].text)
                    Temp.append( soup_detail_line[1].text.split()[0] )
                    Dew.append( soup_detail_line[2 + Col_indadd].text.split()[0])
                    Humidity.append(soup_detail_line[3 + Col_indadd].text.split()[0])
                    Pressure.append(soup_detail_line[4 + Col_indadd].text.split()[0])
                    Visibility.append(soup_detail_line[5+ Col_indadd].text.split()[0])
                    Wind_dir.append(soup_detail_line[6+ Col_indadd].text)
                    Wind_speed.append(soup_detail_line[7+ Col_indadd].text.split()[0])
                    Gust_speed.append(soup_detail_line[8+ Col_indadd].text.split()[0])
                    Event.append(soup_detail_line[10+ Col_indadd].text.strip() )
                    Condition.append(soup_detail_line[11+ Col_indadd].text)
        #%%
    Port = pd.DataFrame(Port, columns=['Port'])    
    Date = pd.DataFrame(Date, columns=['Date'])               
    Time = pd.DataFrame(Time, columns=['Time'])
    Temp = pd.DataFrame(Temp, columns=['Temp'])
    Bodytemp = pd.DataFrame(Bodytemp, columns=['Bodytemp'])
    Dew = pd.DataFrame(Dew, columns=['Dew'])
    Humidity = pd.DataFrame(Humidity, columns=['Humidity'])
    Pressure = pd.DataFrame(Pressure, columns=['Pressure'])
    Visibility = pd.DataFrame(Visibility, columns=['Visibility'])
    Wind_dir = pd.DataFrame(Wind_dir, columns=['Wind_dir'])
    Wind_speed = pd.DataFrame(Wind_speed, columns=['Wind_speed'])
    Gust_speed = pd.DataFrame(Gust_speed, columns=['Gust_speed'])
    Event = pd.DataFrame(Event, columns=['Event'])
    Condition = pd.DataFrame(Condition, columns=['Condition'])
    
    
    sub_result = pd.concat([Port,Date,Time,Temp, Bodytemp, Dew, Humidity,Pressure, Visibility,Wind_dir, Wind_speed, Gust_speed, Event, Condition], axis = 1)

    file_name = 'PORT_'+value.AIRPORT_CODE +'.csv'
    sub_result.to_csv(file_name,index = False)




