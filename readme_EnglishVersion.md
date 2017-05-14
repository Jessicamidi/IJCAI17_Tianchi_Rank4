![图片3.png-376.2kB][1]
# Solution to IJCAI17 Sales Volume Prediction on Koubei Platform
Team: Flamingo 

Ranking list: Rank4

Team member: 
Zhongjie Li, Department of Thermal Engineering, Tsinghua University, lizhongjie1989@163.com

Yichen Yao, Department of Engineering Mechanics, Tsinghua University, yaoyichen@aliyun.com

----------

## 1 Overview
- Background: Alibaba and Ant Financial pile up abundance of online and offline transaction data either from users or from merchants every day. Aimed at providing customized back-end BI (Business Intelligence) service, Ant Financial’s O2O Platform, Koubei, are now focusing on the massive history transaction data to help millions of merchants with transaction statistics, sales analytics and marketing plan suggestions.
- Official website: [Tianchi Platform IJCAI17](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100067.5678.1.amifQx&raceId=231591 "阿里天池 IJCAI17")，4046 teams in total.
- Goal: Koubei provided merchant data, user payment and browse log from 2000 merchants from 2015.07.01 to 2016.10.31. The contestants are supposed to predict the sales volume of each merchant in the next 14 days（2016.11.01-2016.11.14）.
- Evaluation：
<div  align="center"> <img src="http://static.zybuluo.com/Jessy923/k6olhzfz2si5p3n57d5w306x/costF.png" width="650" height="150" alt="Item-based filtering" /></div>
 pit: predicted sales volume of merchant i on day t
 rit: true value


- Notes: The competition encourages data mining (e.g. the use of external data such as weather information).

----------


## 2 Data Mining and Cleaning
### 2.1 Data Mining 
The mining of external data includes weather and holiday information, stored in the folder “additional”, with detailed as follows:
### 2.1.1 Weather data

 - Weather data are from：https://www.wunderground.com， 
which provides detailed weather information, including temperature, dew point, humidity, pressure, visibility, wind speed, instantaneous wind speed, precipitation and other weather conditions around the airport. The historical weather information is sampled at intervals of 30 minutes. 
Here's a glimpse of weather data at Beijing Capital International Airport on November 1, 2016.

<div  align="center"> <img src="http://static.zybuluo.com/Jessy923/ukcrtk5rvuqzvj7sg1qbblky/histGraphAll.gif" /></div>

 - Daily precipitation: sampling interval – day, web crawler - Weather_underground_day.py, generated precipitation table - PRECIP.csv
 - Meteorological conditions series: T sampling interval - 30 minutes, web crawler - Weather_underground_hour.py, generated weather condition table - WEATHER_raw.csv
 - Precipitation index and clear index: the weather conditions are quite complex. We converted the weather parameters into two indicators - precipitation index and clear index. They are shorted for RAIN_IND and CLEAR_IND in the folder “feature / WEATHER_CON_LEVEL.csv”.
 - Due to non-linear relation between human body feeling and meteorological parameters, we add a new feature called [human comfort index SSD][3]
SSD=(1.818t+18.18)(0.88+0.002f)+(t-32)/(45-t)-3.2v+18.2
t: temperature, f: humidity, v: wind speed
 - City weather determination：the city weather is determined by the nearest airport weather information based on latitude and longitude calculation of distance.

### 2.1.2 Holiday data

 - Holiday information - HOLI.csv
 - The label of weekday – 0, weekend – 1 and holiday - 2. Data source comes from the official forum of the competition.


### 2.2 Data cleaning
The data cleaning process includes three parts:  1. Cleaning by rules; 2. Cleaning by pre-training; 3. Only keeps the sales statistics information.

#### 2.2.1 Cleaning by rules

- The raw data contains large amount of purchase within a certain hour for a single user. For instance, the purchase volume of userID 9594359 in the January 30, 2016 in the shopID 878 reaches 209 times. For such situation, the following equation is adopted to eliminate the abnormal consumption.

<div  align="center"> <img src="http://static.zybuluo.com/Jessy923/zul8hn49vki6e1xh8w0mumcd/image_1bg3ktg7ob7s1ggq1omf1f098et9.png"/></div>

<div  align="center"> <img src="http://static.zybuluo.com/Jessy923/mb6ggftge0e5k9yfd3dmvewi/image_1bg3jvnk3d9gbik13fl196j1jb19.png" width="400" height="400"/></div>

- There is a certain amount of start-up time for the newly settled merchants on Koubei platform. The sales interruption phenomenon is also encountered for some merchants, as shown below for shopID 1072. For these situations, data within the opening period (within 7 days) is not used for training, and 1 day before and after the sales interruption period is not used for training.

<div  align="center"> <img src="http://static.zybuluo.com/Jessy923/w0w8yda1orkpiehi26veuikz/image_1bg3k1ncn1t37km5qiq19lm1brnm.png" width="400" height="400"/></div>

- The sales volume is limited by μ ± 2σ in the past 14 days, where μ is the mean of sales volume and σ is the root mean square, as shown below.

<div  align="center"> <img src="http://static.zybuluo.com/Jessy923/wgciv2860j3z1pahbhf7jd59/image_1bg3k26oqng2is11dflrddpuo13.png" width="400" height="400"/></div>


#### 2.2.2 Cleaning by pre-training
See Section 3 for more details. Merchant sales volume can be unpredictably volatile due to promotions, temporary business closure and so on. These factors cannot be easily cleaned by rules. So in this section pre-training is adopted to remove the abnormal values. In the model training process, the model is pre-trained with the under-fitting model and the data with 10% (xgboost1, GBDT) and 25% (xgboost2) of the largest residuals are cleaned.

### 2.3 Only keeps the sales statistics information
Since we only need to predict the daily sales volume of the merchant rather than identify the behavior of the individual user, according to the Law of Large Numbers, we can make the predictions based on the total number of visits and purchases. After the data is cleaned, we preserve the total sales volume by hour and discard the user ID so that the amount of data is now about 1/10 of the original amount.

----------


## 3 Prediction model
![pipeline.JPG-84.2kB][8]
The structure of our solution plan is shown as above. The final sales volume prediction consists of two parts: conventional sales volume in the next 14 days and Double 11 Shopping Festival correction coefficient. Through the double 11 correction coefficient, the sales volume of 2016-11-11,2016-11-12,2016-11-13 is multiplied by coefficient 1.0, 0.2, 0.1, respectively for correction. The training of Double 11 correction coefficient is performed using xgboost model. Merchant information is used as features and the label is double 11 sales increase ratio on the same day for the previous year (2015). For the conventional sales volume prediction, 4 sets of models are adopted, including two sets of xgboost models (different degree of feature processing and data cleaning), a GBDT model and a history mean value model. Detailed model settings are as follows:

### Conventional sales volume prediction

#### Feature and label information


|    Feature and label	   |   instructions    | 
| :-----:   | :-------------:  | 
|History sales volume feature| the sales volume of the past 21 days| 
|Holiday feature	|holiday information of the past 21 days and the 14 predicting days| 
|Weather feature|precipitation, SSD value, precipitation index and clear index of the past 21 days and 4 days around the predicting day|
|Merchant feature|	average ratio of View and Pay, average opening and closing time, business hour, date of initial opening, non-holiday sales median, holiday sales median, holiday / non-holiday sales ratio; business category, per capita consumption, rating, comments, store grade| 
|Label|sales volume for the next 14 days|

#### Training methods
- A total of 481143 valid training samples were generated for the time series of 2000 merchant sales, and 468535 samples were retained after removing data near sales interruption periods and outliers.
- We used two rounds of training. In the first round we used an under-fitting model with the maximum depth of 3 to further clean dirty data. Xgboost and sklearn GBDT were adopted with some basic parameters summarized as follows:
XGBoost-Round_1: log processing for daily sales volume, pre-training sample retention rate is 90%.
XGBoost-Round_2: log processing for daily sales volume, dimensionless by the median of the past three weeks, pre-training sample retention rate is 75%.

|XGBoost	|objective	|max_depth |	learning_rate	|n_estimators|	reg_alpha|	reg_lambda|
|:---:	|:---:	|:---: |:---:	|:---:|:---:|:---:|
|Round_1|	reg:linear|	3|	0.1	|500	|0	|1|
|Round_2	|reg:linear|	5|	0.03|	1600|	1|	0|
GBDT: the retention rate for the first round of training is 90%.

|GBDT|	loss|	max_depth|	learning_rate|	n_estimators|	alpha|
|:---:	|:---:	|:---: |:---:	|:---:|:---:|
|Round_1|lad	|3|	0.1	|500	|0.95|
|Round_2|	lad	|5|	0.1	|500|	0.95|


### History mean value model

- Input: history sales volume for the past 21 days, sales correlation matrix in the last three weeks.
- Output: sales volume in next two weeks and its corresponding confidence in the model fusion.
- Method: the past 21 days are averaged by weekday. Based on the median and average of sales volume in the past three weeks, we used linear fitting to get sales increase. The predicting sales volume in the next two weeks were obtained by the superposition between historical average sales volume and sales volume increase. 
- As the method looks for historically similar (over the past three weeks of correlation) sales curve as a future prediction, it is essentially a combination of the mean value model and KNN method.
- Confidence is the fusion coefficient, only when the correlation coefficient of the three weeks or the correlation coefficient of the latter two weeks is greater than 0.7. The maximum ratio of the mean model is 0.75. The fusion coefficient is calculated as follows:

<div  align="center"> <img src="http://static.zybuluo.com/Jessy923/gxdn8nohm2qsvayrgri4hbh3/eq1png.png" width="300" height="60"/></div>


### Double 11 sales volume correction model
 - Model overview: The time period to be predicted (November 1 to November 14) contains Double 11 festivals. It is seen from the sales figures of many merchants that there is an obvious fluctuation on November 11. The reason may be due to the impact of the online store promotion and the promotion of the catering industry as a bachelor’s festival. However, only about 1/3 of the merchants have sales record of in November 11, 2015. We need to predict the sales volume of these merchants based on other merchants’ sales volume.
 - Features: only include merchant information, including average ratio of View and Pay, average opening and closing time, business hour, date of initial opening, non-holiday sales median, holiday sales median, holiday / non-holiday sales ratio; business category, per capita consumption, rating, comments, store grade
Double 11 sales increase: the weighted sales ratio between the sales volume on Double 11 Day V1111 in the previous year (2015) and the corresponding days two weeks before and two weeks after Double 11 Day V1028, V1104, V1118, V1125. The weight coefficient were 0.15, 0.35, 0.35, 0.15.

<div  align="center"> <img src="http://static.zybuluo.com/Jessy923/5iyvh7olsr32dncmn49vuilo/eq2.png"/></div>
 
-  Training methods: we use xgboost model for training. In order to prevent over-fitting, the parameter settings are more conservative, the maximum depth is 2, and a larger regular term is added. The main settings were as follows: max_depth = 2, learning_rate = 0.01, n_estimators = 500, reg_alpha = 10, gamma = 1

### Model fusion
1.	Fusion of multiple sets of gradient boosting results 
Xgboost1, xgboost2, GBDT results were fused by the ratio of 0.47, 0.34, 0.19.
2.	Fusion of gradient boosting and mean value model 
The results of the mean value model were fused with the results of gradient boosting in step 1, and the fusion coefficient of the mean value model is the confidence obtained by the correlation.
3.	Double 11 coefficient for volume modulation
The sales volume were obtained by the multiplication of Double 11 sales volume and the Double 11 sales volume correction. As 11-12, 11-13 are Saturday and Sunday, we believe that the sales volume on  11-12 and 11-13 is similar to that on 11-11 (Friday). Thus they were multiplied by 0.2 and 0.1 times of sales volume increase coefficient.


## Code instructions
**Step1**：enerate simplified version of user_pay, user_view tables

    data_new/table_regenerate.py

**Step2**：Data mining

    additional/Weather_underground_day.py
    additional/Weather_underground_hour.py

**Step3**：Feature engineering

    feature/ WEATHER_FEATURES.py    
    feature/ SHOP_FEATURES.py
    feature/ TEST_SELLS.py
    feature/FEATURE_MERGE.py

**Step4**：Conventional sales volume model training

    model/xgb_model1.py，model/xgb_model2.py，model/ GBDT_model.py
    model/correlation_model.py

**Step5**：Double 11 correction coefficient training

    model/ DOU11_model.py

**Step6**：Model fusion

    model/model_blend.py
    

  [1]: http://static.zybuluo.com/Jessy923/8i1cq0e2zdrl90s4bsgto2i2/%E5%9B%BE%E7%89%873.png
  [3]: http://baike.baidu.com/link?url=J2KE6H1F1P_qLZrwU6P9c6sxfVKFpG6ob6Vsk997EDBk8kxCyZuY3r8Tj0CEBXU74DyJ1r8M8N9jn6tvTN2GBAAqoJ7VMgbypwCBYx5x-YkQl-PjZgzYyE6hSE4ylTpfBZZ0tRlDU5NcrckI8KLkzjZcK7O430qi8Jf5I1mZPzW
  [8]: http://static.zybuluo.com/Jessy923/bsw2bmxrm5xx4vmt3tu8pujr/pipeline.JPG
