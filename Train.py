# -*- coding: utf-8 -*-

import os
os.chdir("C:\\kongjy_special\\实习工作内容总结\\201805-201808华泰证券\\Task3\\data_sacve")

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


###########################
###                     ###    
###  交易体量相关         ###
###                     ###    
###########################

def sumV(TransData,number=10):
    
    if number > 10 :
        raise BaseException ("the price number should be less equal than 10!")
        
    AskColumnName = "AskSumV" + str(number)
    BidColumnName = "BidSumV" + str(number)
    
    TransData.loc[:,AskColumnName] = pd.Series([0.0] * len(TransData))
    TransData.loc[:,BidColumnName] = pd.Series([0.0] * len(TransData))
    for i in range(number):
        TransData.loc[:,AskColumnName] += TransData['AskV' + str(i)]
        TransData.loc[:,BidColumnName] += TransData['BidV' + str(i)]


def deltaSumV(TransData,number=10):
    
    ColumnName = "deltaSumV" + str(number)
    AskColumnName = "AskSumV" + str(number)
    BidColumnName = "BidSumV" + str(number)
    
    if AskColumnName not in TransData.columns or BidColumnName not in TransData.columns:
        sumV(TransData,number)
    
    TransData.loc[:,ColumnName] = TransData.loc[:,BidColumnName] \
                                                - TransData.loc[:,AskColumnName]

def speedV(TransData,number=10,period=1):
    
    deltaColumnName = "deltaSumV" + str(number)
    speedColumnName = "speedV" + str(number) 
    speedV = pd.Series(pd.rolling_mean(TransData[deltaColumnName],period))
    TransData.loc[:,speedColumnName] = speedV
    

###########################
###                     ###    
### 订单数据( book)      ###
###                     ###    
###########################

def minmax(OrderData,period):
    pass




###########################
###                     ###    
###  移动平均相关（价格）  ###
###                     ###    
###########################


def SMA(TransData, period=10):
    ''' 简单移动平均 simple moving average '''
    SMA = pd.Series(pd.rolling_mean(TransData['Price'], period))
    TransData['SMA'] = SMA
    return TransData

    
def EMA(TransData, period=10):
    ''' 指数移动平均 Exponential Moving Average '''
    EMA = pd.Series(pd.ewma(TransData['Price'], span = period, min_periods = period ))
    TransData['EMA' + str(period)] = EMA
    return TransData



###########################
###                     ###    
###  物理学分量（趋势）    ###
###                     ###    
###########################

def distance(TransData,period=10):
    ''' distance '''
    distance = pd.Series(TransData['Price'].diff(period))
    TransData['DIS' + str(period)] = distance
    return TransData

def velocity(TransData,period=10):
    if ("VELOC" + str(period)) not in TransData.columns:
        distance(TransData,period)
    TransData['VELOC'+ str(period)] = TransData['DIS' + str(period)] / period
    return TransData

def force(TransData,period=10):
    ''' force  '''
    if ("VELOC" + str(period)) not in TransData.columns:
        velocity(TransData,period)
    TransData['FORCE'+ str(period)] = TransData['VELOC'+str(period)] * TransData['Volume'] 
    return TransData

def momentum(TransData,period=10):
    ''' momentum '''
    if ("FORCE" + str(period)) not in TransData.columns:
        force(TransData,period)
    TransData['MOM'+ str(period)] = TransData['FORCE' + str(period)] * period
    return TransData

def energy(TransData,period=10):
    if ("MOM" + str(period)) not in TransData.columns:
        momentum(TransData)
    TransData['ENERGY'+ str(period)] = TransData['MOM' + str(period)] * TransData['Volume']
    return TransData



###########################
###                     ###    
###  震荡统计相关（稳定）  ###
###                     ###    
###########################
    
def std(TransData,period=10):
    STD = TransData['Price'].rolling(window = period).std()
    TransData['STD'+ str(period)] = STD
    return TransData



###########################
###                     ###    
### Below to be checked ###
###                     ###    
###########################
    


def RSI(TransData, per=3):
    ''' Relative Strength Index 
        to be checked! '''
    series = TransData['Price']
    delta = series.diff()
    uuu = delta * 0
    ddd = uuu.copy()
    i_pos = delta > 0
    i_neg = delta < 0
    uuu[i_pos] = delta[i_pos]
    ddd[i_neg] = delta[i_neg]
    rsi = uuu.ewm(ignore_na=True, span=per, min_periods=per, adjust=True).std() / \
          ddd.ewm(ignore_na=True, span=per, min_periods=per, adjust=True).std()
    TransData["RSI" + str(per)] = pd.Series(100 - 100 / (1 + rsi))
    return TransData

def kaufman(TransData, per=10):
    ''' Kaufman efficiency ratio '''
    series = TransData['Price']
    direction = series.diff(per).abs()
    volatility = series.diff().abs().rolling(window=per, center=False).sum()
    kaufman = direction/volatility
    TransData['kaufman' + str(per)] = kaufman
    return TransData
    
'''
test part
'''


price = pd.read_csv('Tick6.csv')\
                                .rename(columns={"Unnamed: 0":"Timestamp"})
Trans = pd.read_csv('Trans6.csv')\
                                .rename(columns={"Unnamed: 0":"Timestamp"})
TransDropList = ["FunctionCode","Date","Time"]
Trans = Trans.drop(TransDropList,axis=1)
price['MatchItem'] = price['MatchItem'].apply(int)

Trans['Timestamp'] = pd.to_datetime(Trans['Timestamp'])
price['Timestamp'] = pd.to_datetime(price['Timestamp'])

# 删除多余的数据---会出现set不交的情况，有三秒没有交易
for i in range(len(price)):
    if Trans.loc[i,'Timestamp'].hour == 9 and Trans.loc[i,'Timestamp'].minute \
                                <= 30 and Trans.loc[i,'Timestamp'].second < 12:
        continue
    else:
        index0 = i
        break
MatchItem = [index0] + list(price.MatchItem)
for i in range(len(price)):
    index1 = MatchItem[i]
    index2 = MatchItem[i+1]
    Trans.loc[index1:index2-1,'Timestamp'] = price.loc[i,'Timestamp']
Trans = Trans.drop(Trans.index[:index0]).drop(Trans.index[index2:])

#group by数据并且载入
temp = Trans[['Timestamp','Volume']].groupby("Timestamp").agg\
        ({"Volume":[np.mean,np.std,np.size,np.median]}).reset_index()

temp.columns = ['Timestamp','MeanPrice','StdPrice','CountPrice','MedianPrice']

price = pd.merge(price,temp,on=['Timestamp'],how='left')


temp = Trans[['Timestamp','Amount']].groupby("Timestamp").agg\
        ({"Amount":[np.mean,np.min,np.std,np.max,np.size,np.median]}).reset_index()

temp.columns = ['Timestamp','MeanVolume','MinVolume','StdVolume','MaxVolume',\
                                                'CountVolume','MedianVolume']
price = pd.merge(price,temp,on=['Timestamp'],how='left')



sumV(price)
deltaSumV(price)
speedV(price)

SMA(price)
EMA(price)

distance(price)
velocity(price)
force(price)
momentum(price)
energy(price)

std(price)

RSI(price)
kaufman(price)

#清除多余的column
AB = lambda A,B : [str(i)+str(j) for i in A for j in B]
PriceDropList = AB(['AskP','BidP','AskV','BidV'],[i for i in range(1,10)])
PriceDropList += ['Time','MatchItem','Date']
price = price.drop(PriceDropList,axis=1)




###############################################################################
def compa0(x):
    if x > 0 : return 1
    elif x< 0 : return -1
    else :return 0 


price['Future_diff'] = price['Price'].diff().shift(-1)
price['isadd'] = price['Future_diff'].apply(compa0)
#price['Future_mean10'] = price['Price'].rolling(window = 10,min_periods=10).mean().shift(-10)
#price['Future_max10'] = price['Price'].rolling(window = 10,min_periods=10).max().shift(-10)


price = price.drop("Timestamp",axis=1)
droplist = price.index[:MatchItem[0]].join\
                            (price.index[MatchItem[-1]:],how="outer")
price = price.drop(droplist,axis=0)
#price = price.drop(["AskP0","BidP0"],axis=1)
price = price.fillna(0)
column = price.columns

from sklearn import preprocessing
#price = preprocessing.scale(price.iloc[:,:-2])
price = pd.DataFrame(price,columns=column)

from matplotlib import pyplot as plt
plt.figure(1)
plt.plot(price.Future_diff)


# X- Y split and test - train split
from sklearn.model_selection import train_test_split
X = price.loc[:,price.columns[:-2]]
Y = price.loc[:,price.columns[-2:]]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)

split_margin = int(price.shape[0] * 0.85)
X_train = price.loc[:split_margin,price.columns[:-2]]
Y_train = price.loc[:split_margin,price.columns[-2:]]
X_test = price.loc[split_margin:,price.columns[:-2]]
Y_test = price.loc[split_margin:,price.columns[-2:]]
#del price,temp,Trans,MatchItem

###############################################################################
# model visualization
import seaborn as sns
plt.plot(price.iloc[:,-2])
plt.hist(price.iloc[:,-2],bins=10)

    
def plot_scatter(column):
    column_name = price.columns[column]
    temp = price.iloc[:,[-2,column]]
    temp['isdayu0'] = (price.iloc[:,-2]).map(compa0)
    plt.figure()
    plt.scatter(temp.loc[temp['isdayu0'] > 0,column_name],\
                            temp.loc[temp['isdayu0'] > 0,'Future_diff'],c=["r"]
                            ,label="up")
    plt.scatter(temp.loc[temp['isdayu0'] == 0,column_name],\
                            temp.loc[temp['isdayu0'] == 0,'Future_diff'],c=["g"]
                            ,label="equal")
    plt.scatter(temp.loc[temp['isdayu0'] < 0,column_name],\
                            temp.loc[temp['isdayu0'] < 0,'Future_diff'],c=["b"]
                            ,label="down")
    plt.title('%s - price_change'%(column_name))
    plt.xlabel("%s"%(column_name))
    plt.ylabel("price")
    plt.legend()
    plt.grid()
    
def plot_hist(column):
    column_name = price.columns[column]
    temp = price.iloc[:,[-2,column]]
    temp['isdayu0'] = (price.iloc[:,-2]).map(compa0)
    plt.figure()
    plt.hist(x=[temp.loc[temp['isdayu0'] > 0,column_name],\
                temp.loc[temp['isdayu0'] == 0,column_name],\
                temp.loc[temp['isdayu0'] < 0,column_name]],
                stacked=True,color = ['r','g','b'],label=["up","equal","down"])
    plt.title('%s - price_change'%(column_name))
    plt.xlabel("%s"%(column_name))
    plt.ylabel("price")
    plt.legend()

def correlation_headmap():
    colormap = sns.palplot(sns.diverging_palette(240, 10, n=11))
    sns.heatmap(price.iloc[:,:-1].corr(),
            cmap = colormap,
            cbar=True,#whether to add a colorbar
            annot=True,#whether to write the data
            )

def double_scatter(column):
    '''
    read _in list
    '''
    if type(column) is not list:
        raise BaseException ("Please input type list with two columns!")
    plt.figure()
    column_name = price.columns[column]
    temp = price.iloc[:,[-2]+column]
    temp['isdayu0'] = (price.iloc[:,-2]).map(compa0)
    plt.scatter(temp.loc[temp['isdayu0']>0,column_name[0]],\
                temp.loc[temp['isdayu0']>0,column_name[1]],c='r',label='up')
    plt.scatter(temp.loc[temp['isdayu0']==0,column_name[0]],\
                temp.loc[temp['isdayu0']==0,column_name[1]],c='g',label='equal')
    plt.scatter(temp.loc[temp['isdayu0']<0,column_name[0]],\
                temp.loc[temp['isdayu0']<0,column_name[1]],c='b',label='down')    
    plt.title('%s - %s' %(column_name[0],column_name[1]))
    plt.xlabel('%s'%(column_name[0]))
    plt.ylabel('%s'%(column_name[1]))
    plt.legend()

    
#plot_scatter(25)
#plot_hist(31)
#correlation_headmap()
#double_scatter([27,31])



###############################################################################
# feature selection ---  

from sklearn.cross_validation import cross_val_score,ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
import time


#  Embedded
clf = RandomForestRegressor(n_estimators=20,max_depth=10)
scores = {}
for i in range(X_train.shape[1]):
    score = cross_val_score(clf,X_train.iloc[:,i:i+1].values,Y_train.Future_diff\
                    .values,scoring='r2',cv=ShuffleSplit(len(X_train), 3, .3))
    scores[X_train.columns[i]] = round(np.mean(score), 3)
scores = sorted(scores.items(),key=lambda x:x[1])


#  包裹  wrapper  
from sklearn.feature_selection import RFE
#X_train_new = RFE(estimator=clf,n_features_to_select=2).\
#                 fit_transform(X_train.iloc[:,:4].values,Y_train.iloc[:,1].values)
time1 = time.strftime("%Y-%m-%d %H:%M:%S")
print (time1)


###############################################################################
# model part
Y_train1 = Y_train.iloc[:,0]
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error,confusion_matrix,\
                            precision_score,recall_score

kf = KFold(n_splits = 6)
for train_index, test_index in kf.split(X_train.values):
    xgb_model = xgb.XGBRegressor().fit(X_train.values[train_index],\
                                                Y_train1.values[train_index])
    predictions = xgb_model.predict(X_train.values[test_index])
    realdata = Y_train1.values[test_index]


predictReal = xgb_model.predict(X_test.values)

Y_test = Y_test.iloc[:,0]
print ("mean square error is : ",mean_squared_error(predictReal,Y_test.values))
print ("confusion matrix is :",confusion_matrix(predictReal>0,Y_test.values>0))
print ("Precision_score is :",precision_score(predictReal>0,Y_test.values>0))
#print ("Recall_score is :",recall_score(predictReal>0,Y_test.values>0))

predictReal_Data = pd.Series(predictReal>0).map({True:1,False:0})
Y_test_Data = (Y_test>0).map({True:1,False:0})

print ("confusion matrix is :",confusion_matrix(predictReal_Data,Y_test_Data))
print ("Precision_score is :",precision_score(predictReal_Data,Y_test_Data))
#print ("Recall_score is :",recall_score(predictReal_Data,Y_test_Data))

