# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import pandas as pd


Name = lambda x,y:x+str(y)
NameList = lambda x,y: [Name(x,i) for i in y]

def _compa0(x):
    if x > 0 : return 2
    elif x< 0 : return 0
    elif x==0 :return 1

def smooth(temp):
    '''中位数平滑后，用此方法去除剩余的3tick不平滑点
        temp 是median平滑后的数据'''
    abnormal_index = []
    for i in range(1,len(temp) - 1):
        thisdiff = temp[i] - temp[i-1]
        nextdiff = temp[i+1] - temp[i]
        if  thisdiff * nextdiff < 0:
            abnormal_index.append(i)
            temp[i] = temp[i] - thisdiff
#    print ("totally %s abnormal points fixed!"%(len(abnormal_index)))
    return temp,abnormal_index

def scatter_classification(data,column_name,yname):
    temp = data.loc[:,[column_name,yname]]
    temp['isdayu0'] = (data.loc[:,column_name]).map(_compa0)
    plt.figure()
    plt.plot(temp[yname],label='median_5',linewidth=3,c='b')
    plt.scatter(x=temp.loc[temp['isdayu0'] > 1,yname].index,\
                y=temp.loc[temp['isdayu0'] > 1,yname],\
                c=["r"],marker='^',label="up",linewidths=10)
    plt.scatter(x=temp.loc[temp['isdayu0'] == 1,yname].index,\
                y=temp.loc[temp['isdayu0'] == 1,yname],\
                c=["k"],marker='o',label="equal",linewidths=10)
    plt.scatter(x=temp.loc[temp['isdayu0'] < 1,yname].index,\
                y=temp.loc[temp['isdayu0'] < 1,yname],\
                c=["g"],marker='v',label="down",linewidth=10)
    plt.title('%s - self.Tick_change'%(column_name))
    plt.xlabel("%s"%(column_name))
    plt.ylabel("TickData")
    plt.legend()
    plt.grid()
    
def trinary(x):
    '''-1 0 1 映射到 0 1 2'''
    if   x < 0    :        return 0
    elif x > 0    :        return 2
    elif x==0 : return 1

def cumsum(tempTick):
    '''    计算每个10档价格出的cumsum Volume    '''
    AskSumName = NameList('cumSumAskV',range(10))
    BidSumName = NameList('cumSumBidV',range(10))
    AskName = NameList('AskV',range(10))
    BidName = NameList('BidV',range(10))
    tempTick[AskSumName] = tempTick[AskName].apply(pd.np.cumsum,axis=1)
    tempTick[BidSumName] = tempTick[BidName].apply(pd.np.cumsum,axis=1)


def findequal(tempTick,hands):
    '''    找到cumsum价格中的等量的处的价格    '''
    AskSumName = NameList('cumSumAskV',range(10))
    BidSumName = NameList('cumSumBidV',range(10))
    tempTick['Askindex'+str(hands)] = tempTick[AskSumName].apply\
                    (lambda x:x[x<1e2 * hands].shape[0],axis=1)
    tempTick['Bidindex'+str(hands)] = tempTick[BidSumName].apply\
                    (lambda x:x[x<1e2 * hands].shape[0],axis=1) 
    tempTick['AskPriceV'+str(hands)] = tempTick.apply\
        (lambda x:x['AskP'+str(int(pd.np.clip(x['Askindex'+str(hands)],0,9)))],axis=1)
    tempTick['BidPriceV'+str(hands)] = tempTick.apply\
        (lambda x:x['BidP'+str(int(pd.np.clip(x['Bidindex'+str(hands)],0,9)))],axis=1)

def BidAskPower(tempTick,BS):
    '''    等量强度因子    '''
    fixed = (BS == "Bid")*(-1) + (BS == "Ask") *1
    tempTick[BS+'d01'] = tempTick[BS +'PriceV5'] - tempTick['Price']
    tempTick[BS+'d11'] = tempTick[BS +'PriceV10'] -tempTick[BS +'PriceV5']
    tempTick[BS+'d21'] = tempTick[BS +'PriceV15'] - tempTick[BS +'PriceV10']
    tempTick[BS+'d31'] = tempTick[BS +'PriceV20'] - tempTick[BS +'PriceV15']
    tempTick[BS + 'Strength'] = (tempTick[BS+'d01'] * 7 + tempTick[BS+'d11'] * 5 \
            + tempTick[BS+'d21'] * 3  + tempTick[BS + 'd31'] )/16 * fixed

def BidAskPower2(tempTick,BS):
    '''    Tick强度因子    '''
    fixed = (BS == "Bid")*(-1) + (BS == "Ask") *1
    tempTick[BS+'d02'] = tempTick[BS +'P1'] - tempTick['Price']
    tempTick[BS+'d12'] = tempTick[BS +'P2'] -tempTick[BS +'P1']
    tempTick[BS+'d22'] = tempTick[BS +'P3'] - tempTick[BS +'P2']
    tempTick[BS+'d32'] = tempTick[BS +'P4'] - tempTick[BS +'P3']
    tempTick[BS + 'Strength2'] = (tempTick[BS+'d02'] * 7 + tempTick[BS+'d12'] * 5 \
            + tempTick[BS+'d22'] * 3 + tempTick[BS+'d32'] )/16 * fixed

def BidAskTime(tempTick,BS,period):
    time_weights = [pow(2,i) for i in range(period)]
    fixed = (BS == "Bid")*(-1) + (BS == "Ask") *1
    tempTick[BS+'d01'+str(period)] = tempTick[BS+'d01'].diff().rolling(window=\
             period).apply(lambda x:pd.np.multiply(x,time_weights).sum())
    tempTick[BS+'d11'+str(period)] = tempTick[BS+'d11'].diff().rolling(window=\
             period).apply(lambda x:pd.np.multiply(x,time_weights).sum())
    tempTick[BS+'d21'+str(period)] = tempTick[BS+'d21'].diff().rolling(window=\
             period).apply(lambda x:pd.np.multiply(x,time_weights).sum())
    tempTick[BS+'d31'+str(period)] = tempTick[BS+'d31'].diff().rolling(window=\
             period).apply(lambda x:pd.np.multiply(x,time_weights).sum())
    tempTick[BS + 'Strength'+str(period)] = (tempTick[BS+'d01'+str(period)] * 7 \
            + tempTick[BS+'d11'+str(period)] * 5 \
            + tempTick[BS+'d21'+str(period)] * 3 \
            + tempTick[BS+'d31'+str(period)] )/16 * fixed
    tempTick[BS + 'Strength'+str(period)] = tempTick[BS + 'Strength'+str(period)].fillna(0)

def BidAskTime2(tempTick,BS,period):
    time_weights = [pow(2,i) for i in range(period)]
    fixed = (BS == "Bid")*(-1) + (BS == "Ask") *1
    tempTick[BS+'d02'+str(period)] = tempTick[BS+'d02'].diff().rolling(window=\
             period).apply(lambda x:pd.np.multiply(x,time_weights).sum())
    tempTick[BS+'d12'+str(period)] = tempTick[BS+'d12'].diff().rolling(window=\
             period).apply(lambda x:pd.np.multiply(x,time_weights).sum())
    tempTick[BS+'d22'+str(period)] = tempTick[BS+'d22'].diff().rolling(window=\
             period).apply(lambda x:pd.np.multiply(x,time_weights).sum())
    tempTick[BS+'d32'+str(period)] = tempTick[BS+'d32'].diff().rolling(window=\
             period).apply(lambda x:pd.np.multiply(x,time_weights).sum())
    tempTick[BS + 'Strength2'+str(period)] = (tempTick[BS+'d02'+str(period)] * 7 \
            + tempTick[BS+'d12'+str(period)] * 5 \
            + tempTick[BS+'d22'+str(period)] * 3 \
            + tempTick[BS+'d32'+str(period)] )/16 * fixed
    tempTick[BS + 'Strength2'+str(period)] = tempTick[BS + 'Strength'+str(period)].fillna(0)