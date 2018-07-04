# -*- coding: utf-8 -*-
import Price
import numpy as np
import pandas as pd

'''
价格变化 ：
        1 Tick 价格相减  PLUSDM(高价)、MINUSDM(低价)、TR(波幅)、MTM(价格)--10****1参数可调
        1 Tick 高低价格变动 PLUSDI MINUSDI DX ----------- 共同更改周期
        以上指标的平均，ADX= mean(DX,t) , ADXR = mean(DX,DXt) -- 10 **** 1可调

        APO ： 长期的EMA - 短期的EMA -- 12,26 **** 2可调
        AROON ： 周期内上一个最点距离现在最大点的周期 -- 60 **** 1可调
        AROONOSC : AROONUP - AROONDOWN -- 60 **** 1可调
        BBI ： 四条移动均线加权平均  -- 3 6 12 24 **** 4可调
        PBX ： 三条移动均线平均  -- 10 10*2 10*4  **** 1可调
        CMO ： 价格增长Ui、减小Di。mean(Ui,t1),mean(Di,t1)。 相减/相加 -- 10 ***** 1可调
        DMA ： DMA=SMA(n1)-SMA(n2), AMA=mean(DMA,n3) -- 5 10 10 **** 3可调
        MACD :  DIF--EMA(t1)-EMA(t2)  DEA--ewma(DIF,t3) t1=12,t2=26,t3=10,MACD--DIF-DEA ** 3可调
        TRIX : EMA(10) , EWMA(EMA,10), EWMA(EWMA2,10) delta(EMA3)/EMA3 ** 1可调


'''

def ff(x):
    if x <0:
        return 0
    elif x>1:
        return 1
    else:
        return x

def PLUSDM(data):
    '''
    正向动量 : 高价变化
    '''
    data['PLUSDM'] = (data['HighPrice'] - data['HighPrice'].shift(1)).map(ff)

def MINUSDM(data):
    '''
    负向动量 : 低价变化
    '''
    data['MINUSDM'] = (data['LowPrice'] - data['LowPrice'].shift(1)).map(ff)


def TR(data):
    '''
    真实波幅
    '''
    data['tempHP'] = data['HighPrice'].shift(1)
    data['tempLP'] = data['LowPrice'].shift(1)
    data['TR'] = data[['AskP0','tempHP','tempLP']].apply(lambda x:\
        max(x[0],x[1]) - min(x[0],x[2]),axis=1)
    data.drop(['tempHP','tempLP'],axis=1)


def PLUSDI(data, period=10):
    '''
    正向指标
    '''
    data['PLUSDI'+str(period)] = data['PLUSDM'].rolling\
                        (window=period,min_periods=period).sum()\
                        *100.0 / (data['TR'].rolling\
                        (window=period,min_periods=period).sum()+1)

def MINUSDI(data, period=10):
    data['MINUSDI'+str(period)] = data['MINUSDM'].rolling\
                        (window=period,min_periods=period).sum()\
                        *100.0 / (data['TR'].rolling\
                        (window=period,min_periods=period).sum()+1)

def DX(data, period=10):
    '''
    动向指标
    '''
    data['DX'+str(period)] =    ( data['PLUSDI'+str(period)] - data['MINUSDI'+str(period)] )/\
                    ( data['PLUSDI'+str(period)] + data['MINUSDI'+str(period)])


def ADX(data, period=10):
    ######
    '''
    平均动量指数
    '''
    if ("DX"+str(period)) not in data.columns:
        DX(data,period)
    data['ADX'+str(period)] = data['DX'+str(period)].rolling(window=period,min_periods=period).mean()

def ADXR(data, period=10):
    '''
    平均动量指数评估指标
    '''
    data['tempADX'] = data['ADX'+str(period)].shift(period)
    data['tempADX'] = data['tempADX'].fillna(data.loc[0,'ADX'+str(period)])
    data['ADXR'+str(period)] = (data['tempADX'] + data['ADX'+str(period)] ) / 2.0
    data = data.drop(['tempADX'], axis=1)

def AMV(data, period=10):
    '''
    成本均价线，此处类似于amount，pass掉
    '''
    pass

def APO(data, short_period=12, long_period=26):
    '''
    绝对价格震荡指标，长期的EMA减去短期的EMA
    '''
    Price.EMA(data,short_period)
    Price.EMA(data,long_period)
    data['APO'+str(short_period)+str(long_period)] = data['EMA'+str(long_period)] - data['EMA'+str(short_period)]

def ASI(data):
    '''
    震动升降指标
    '''
    pass

def AROON(data,period=30):
    '''
    ARNOON指标，关于开盘、最高、最低、收盘的数据，高频不行 ~~~
    '''
    # 玄学，np.argwhere(x==np.max(x)) 已经是numpy格式了，自测可行
    data['AROONUP'+str(period)] = data['AskP0'].rolling(window=period).apply(lambda x:\
            (1.0*period-1 - np.argwhere(x == np.max(x)).ravel()[0])/period)
    data['AROONDOWN'+str(period)] = data['AskP0'].rolling(window=period).apply(lambda x:\
            (1.0*period-1 - np.argwhere(x == np.min(x)).ravel()[0])/period)

def AROONOSC(data,period=30):
    '''
    AROON震荡指标
    '''
    if ("AROONUP" + str(period)) not in data.columns:
        AROON(data,period)
    data['AROONOSC'+str(period)] = data['AROONUP'+str(period)] - data['AROONDOWN'+str(period)]

def BBI(data,n1=3,n2=6,n3=12,n4=24):
    '''
    多空指数，四条移动均线的加权平均
    '''
    Price.SMA(data,n1)
    Price.SMA(data,n2)
    Price.SMA(data,n3)
    Price.SMA(data,n4)
    y = lambda x:"SMA" + str(x)
    data["BBI"+str(n1)+str(n2)+str(n3)+str(n4)] = (data[y(n1)] + \
                               data[y(n2)]+data[y(n3)]+data[y(n4)])/4.0
    data = data.drop([y(n1),y(n2),y(n3),y(n4)],axis=1)

def BOP(data):
    '''
    力量均衡指标
    '''
#    data["BOP"] = 1.0 * ( data['AskP0'] - data['AskP0'].shift(1) ) / \
#                        ( data['HighPrice'] - data['LowPrice'] )
    pass

def CMO(data, period=10):
    '''
    钱德震荡动量
    '''
    data['Ui'+str(period)] = data['AskP0'].diff().apply(lambda x:(x>0)*x)
    data['Di'+str(period)] = data['AskP0'].diff().apply(lambda x:(x<0)*(-x))
    data['RUi'+str(period)] = data['Ui'+str(period)].rolling(window=period,min_periods=1).sum()
    data['DUi'+str(period)] = data['Di'+str(period)].rolling(window=period,min_periods=1).sum()
    data['CMO'+str(period)] = 1.0 * (data['RUi'+str(period)] - data['DUi'+str(period)]) / \
                (data['RUi'+str(period)] + data['DUi'+str(period)]) * 100
    data.drop(['Ui'+str(period),'Di'+str(period)],axis=1)

def DMA(data, n1=5, n2=10, period=10):
    '''
    平均线差
    '''
    Price.SMA(data,n1)
    Price.SMA(data,n2)
    data['DMA'+str(n1)+str(n2)] = data['SMA'+str(n1)] - data['SMA'+str(n2)]
    data['AMA'+str(n1)+str(n2)+str(period)] = data['DMA'+str(n1)+str(n2)].rolling(window=period).mean()

def ENV(data):
    '''
    轨道线，上下1%  无意义
    '''
    pass

def LWR(data):
    '''
    Williams %R 威廉 指标，关于Tick内的高低价格，无意义
    '''
    pass

def MACD(data,short_period=12,long_period=26, dea_period=10):
    '''
    MACD 波动太大，统计意义不明显-
    '''
    name_short = "EMA" + str(short_period)
    name_long = "EMA" + str(long_period)
    Price.EMA(data,short_period)
    Price.EMA(data,long_period)
    DIFName = 'DIF'+str(short_period) + str(long_period)
    DEAName = 'DEA'+str(short_period) + str(long_period)+str(dea_period)
    MACDName = 'MACD'+str(short_period) + str(long_period)+str(dea_period)
    data[DIFName] = data[name_short] - data[name_long]
    data[DEAName] = pd.ewma(data[DIFName],span=dea_period, min_periods=1)
    data[MACDName] = ( data[DIFName] - data[DEAName] ) * 2.0

def TRIX(data,period=10):
    '''
    三重指数平滑移动平均线
    '''
    Price.EMA(data,period)
    EMA2 = pd.Series(pd.ewma(data['EMA'+str(period)],span=period,min_periods=1))
    EMA3= pd.Series(pd.ewma(EMA2,span=period,min_periods=1))
    data['TRIX'+str(period)] = (1.0* EMA3 - EMA3.shift(1))/EMA3

def MTM(data,period=10):
    '''
    所谓动量，价格之差
    '''
    data['MTM'+str(period)] = data['AskP0'].diff(period)

def PBX(data,period=10):
    '''
    瀑布线
    '''
    Price.SMA(data,period)
    Price.SMA(data,2*period)
    Price.SMA(data,4*period)
    data['PBX'+str(period)] = (data['SMA'+str(period)] + data['SMA'+str(2*period)] \
                    + data['SMA'+str(4*period)]) /3.0
    data = data.drop(['SMA'+str(2*period),'SMA'+str(4*period)],axis=1)

def ULTOSC(data):
    '''
    终极震荡指标
    '''
    pass

def UOS(data):
    '''
    终极震荡
    '''
    pass
