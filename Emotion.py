# -*- coding: utf-8 -*-
'''
ARBR  : 高价差/低价差  -- 10 ***** 1参数可调
PSY : 周期内的涨跌占比 -- 10 **** 1参数可调
VR  : 上涨的单/下跌单子 -- 10 **** 1参数可调
'''


def ARBR(data,period=10):
    '''
    AR 是开盘，BR是结束，由于时间段就选一个
    '''

    HC = data['HighPrice'] - data['Price']
    CL = data['Price'] - data['LowPrice']
    data['BR'+ str(period)] = (HC.rolling(window=period).sum() + 1) / \
                              (CL.rolling(window=period).sum() + 1)

def CR(data, period=10):
    '''
    中间意愿指标  周期太短，和上面差不多
    '''
    pass

def PSY(data, period=10):
    '''
    统计周期内涨的占比
    '''
    tempdata = (data['AskP0'] - data['AskP0'].shift(1)) > 0
    data['PSY'+str(period)] = tempdata.rolling(window=period).sum() * 1.0 / period


def VR(data, period=10):
    '''
    成交量变异率指标
    '''
    def f(x):
        if      x<0: return 0
        elif    x>0: return 1
        else       : return 1/2

    data['tempU'+str(period)]=(data['Price']-data['Price'].shift(1)).apply(f)*data['Volume'] + 1.0
    data['tempD'+str(period)]=(data['Price'].shift(1)-data['Price']).apply(f)*data['Volume'] + 1.0
    data['VR'+str(period)] = (data['tempU'+str(period)].rolling(window=period).sum() * 1.0 ) \
                /(data['tempD'+str(period)].rolling(window=period).sum() )
