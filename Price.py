# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

'''
SMA -- 10 **** 1参数可调
EMA -- 10 **** 1参数可调
DEMA -- 10,1.0 **** 2参数可调
KAMA 区间变动的价格占比-- 10 **** 1参数可调
STDDEV -- 10 ****1参数可调
T3 TMA WMA  -- 10  **** 1参数可调
'''


def TYPPRICE(data):
    '''
    典型价格，高、低、价格
    '''
    data['TYPPRICE'] = (data['HighPrice']+data['LowPrice']+data['AskP0'])/3.0


def WCLPRICE(data):
    data['WCLPRICE']=(data['HighPrice']+data['LowPrice']+data['AskP0']*2.0)/4.0

def SMA(data, period=10):
    '''
    简单移动平均 simple moving average
    '''
    name = "SMA" + str(period)
    data[name] = data['AskP0'].rolling(window=period,min_periods=1).mean()


def EMA(data, period=10):
    '''
    指数移动平均 Exponential Moving Average
    '''
    name = "EMA" + str(period)
    data[name] = pd.Series(pd.ewma(data['AskP0'],
                                    span = period, min_periods = 1 ))

def DEMA(data, period = 10, v_factor = 1.0):
    '''
    双指数移动均线
    '''
    nameDEMA = "DEMA" + str(period)
    nameEMA = "EMA" + str(period)
    if nameEMA not in data.columns:
        EMA(data,period)
    data[nameDEMA] = (1+v_factor)* data[nameEMA] - pd.Series(pd.ewma\
                (data[nameEMA], span = period, min_periods = 1 )) * v_factor

def KAMA(data, period=10):
    '''
    Kaufman efficiency ratio
    '''
    direction = data['AskP0'].diff(period).abs()
    volatility = data['AskP0'].diff().abs().rolling\
                                        (window=period, center=False).sum()
    kaufman = direction * 1.0 / volatility
    data['kaufman' + str(period)] = kaufman

def STDDEV(data, period=10):
    '''
    标准差
    '''
    nameSMA = "SMA" + str(period)
    if nameSMA not in data.columns:
        SMA(data,period)
    data['STDDEV'+str(period)] = data[nameSMA].rolling(window = period,min_periods=1).std()


def T3(data, period=10, v_factor=0.7):
    '''
    双指数移动均线改进  和 TEMA
    '''
    nameDEMA = "DEMA" + str(period)
    nameT2 = "T2" + str(period)
    nameT3 = "T3" + str(period)

    if nameDEMA not in data.columns:
        DEMA(data,period)

    data[nameT2] = (1+v_factor)* data[nameDEMA] - pd.Series(pd.\
             ewma(data[nameDEMA], span = period, min_periods = 1 )) * v_factor
    data[nameT3] = (1+v_factor)* data[nameT2] - pd.Series(pd.\
             ewma(data[nameT2], span = period, min_periods = 1 )) * v_factor

    data['TEMA' + str(period)] = 3*data[nameDEMA]-3*data[nameT2]+data[nameT3]

def TMA(data, period=10):
    '''
    三角线
    '''
    SMA(data,int(period/2))
    data['TMA'+str(period)] = data['SMA'+str(int(period/2))].rolling(window=period+1\
                                                    ,min_periods=1).mean()

def WMA(data, period=10):
    '''
    加权移动平均线
    '''
    suma = sum(range(period+1))
    suma_list = [i/suma for i in range(period)]
    data['WMA'+str(period)] = data['AskP0'].rolling(window=period,min_periods=period).\
            apply(lambda x:sum(np.multiply(x,suma_list)))


def AVGPRICE(data):
    pass

def MEDPRICE(data):
    pass

def MIDPOINT(data):
    pass

def MIDPRICE(data):
    pass

def SMA2(data):
    pass

def TEMA(data):
    pass

def TR(data):
    pass

def TRMA(data):
    pass

def VMA(data):
    pass
