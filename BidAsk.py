# -*- coding: utf-8 -*-

import Price
import numpy as np
import pandas as pd

'''
BIAS : (y-SMA(t))/y -- 10 **** 1参数可调
ROC  : (y.diff(1))/y -- 3 **** 1参数可调
ROCR : (y)y.shift(i) -- 3 **** 1参数可调
CCI  : (y-SMA)/std -- 10 **** 1参数可调
KDJ  : 价格区间变化减去max min变化， -- 10 **** 1参数可调
MFI  : 区间资金流入/(流入+流出) -- 10 **** 1参数可调
PPO  ： (EMAt - EMAt2)/EMAt  -- 20 10 **** 2参数可调
RSI  ： 区间涨/区间跌的  -- 10 **** 1参数可调
'''



def ATR(data):
    '''
    平均真实振幅
    '''
    pass

def BIAS(data, period=10):
    '''
    乖离率,偏离涨幅
    '''
    nameSMA = "SMA" + str(period)
    if nameSMA not in data.columns:
        Price.SMA(data, period)
    data["BIAS"+str(period)] = (1.0*data['AskP0'] - data[nameSMA])/data[nameSMA]


def CCI(data, period=10):
    '''
    自己定义的CCI，价格偏离/方差
    '''
    nameSMA = "SMA" + str(period)
    nameSTD = "STDDEV" + str(period)
    if nameSMA not in data.columns:
        Price.SMA(data,period)
    if nameSTD not in data.columns:
        Price.STDDEV(data,period)
    data['CCI'+str(period)] = (1.0 * data['AskP0'] - data[nameSMA]) /(data[nameSTD] + data[nameSTD].mean())
    data['CCI'+str(period)] = data['CCI'+str(period)].fillna(0) / data['CCI' + str(period)].mean()
    data = data.drop(['SMA'+str(period)],axis=1)

def DPO(data):
    '''
    偏离涨幅
    '''
    pass

def KDJ(data, period=20):
    '''
    随机指标
    '''
    listc = [1/3*pow(2/3,i) for i in range(period)][::-1]

    MAXLP = data['AskP0'].rolling(window=period).max()
    MINLP = data['LowPrice'].rolling(window=period).min()
    MAXHP = data['HighPrice'].rolling(window=period).max()
    data['RSV'+str(period)] = (1.0*data['AskP0'] - MAXLP) / (MAXHP - MINLP + 1e-4)

    data['KT'+str(period)] = data['RSV'+str(period)].rolling(window=period).apply(lambda x:\
                        sum(np.multiply(x,listc)))
    data['DT'+str(period)] = data['KT'+str(period)].rolling(window=period).apply(lambda x:\
                        sum(np.multiply(x,listc)))
    data['JT'+str(period)] = 3*data['DT'+str(period)] - 2 * data['KT'+str(period)]

def MFI(data, period=10):
    '''
    资金流向
    '''
    def clip(x):
        if x < 0:
            return 0
        return x

    data['diffdata'] = data['Price'].shift(-1)
    data['PMF'+str(period)] = data[['Price','diffdata','Volume']].apply \
                        (lambda x:(x[0]>x[1])*x[0]*x[2],axis=1)
    data['NMF'+str(period)] = data[['Price','diffdata','Volume']].apply \
                        (lambda x:(x[0]<x[1])*x[0]*x[2],axis=1)
    data['MFI'+str(period)] = (data['PMF'+str(period)].rolling(window=period).sum() / \
                ((data['PMF'+str(period)]+data['NMF'+str(period)]).rolling(window=period).sum()+1e-6))\
                .apply(clip)
#    data = data.drop(['diffdata'],axis=1)
    del data['diffdata']

def NATR(data, period=3):
    pass


def OSC(data):
    '''
    当日收盘与几日均价比较
    '''
    pass

def PPO(data, long_period=20, short_period=10):
    '''
    价格震荡百分比
    '''
    Price.EMA(data,long_period)
    Price.EMA(data,short_period)
    data['PPO'+str(long_period)+str(short_period)] = (data['EMA'+str(long_period)] - data['EMA'+str(\
                short_period)] ) / data['EMA'+str( short_period)]


def ROC(data, period = 3):
    '''
    变动率
    '''
    data['ROC' + str(period)] = data['AskP0'].diff(period) / data['AskP0']

def ROCP(data):
    '''
    与ROC一样，100倍数关系
    '''
    pass

def ROCR(data, period=3):
    '''
    y/yi-1
    '''
    data['ROCR'+str(period)] = data['AskP0'] / data['AskP0'].shift(period)

def ROCR100(data):
    pass

def RSI(data,period=10):
    U = data['AskP0'].diff().apply(lambda x:max(x,0))
    D = data['AskP0'].diff().apply(lambda x:max(-x,0))
    RS = pd.ewma(U,span=period, min_periods=period)*1.0 / \
                        pd.ewma(D, span=period, min_periods=period)
    data['RSI'+str(period)] = 100 - 100.0/(1+RS)

def WilliamR(data):
    '''
    等同于KDJ种的RSV值
    '''
    pass
