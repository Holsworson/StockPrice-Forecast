# -*- coding: utf-8 -*-

'''
数据采样
'''
import pandas as pd

class Sample(object):
    
    def __init__(self,Tick):
        self.Tick = Tick
    
    def sample(self):
        self.Tick['SMA_median'] = self.Tick['median_5'].rolling(window=5,min_periods=1).mean().shift(-2)
        self.Tick['diff1_EMAmedian'] = self.Tick['SMA_median'].diff()
        self.Tick['former1'] = self.Tick['diff1_EMAmedian'].rolling(window=2).\
                                apply(lambda x:pd.np.cumprod(x)[-1]).shift(-1)
            # 去除一阶残差变号的点
        def threecom(x):
            if (x[0] == 0) and (x[1]==0) and (x[2]==0):
                return 0
            if (x[0]>=0) and (x[1]<= 0) and (x[2]>=0):
                return 1
            else:
                return 0
        self.Tick['is_transi'] = self.Tick['former1'].rolling(window=3).\
                                                    apply(threecom).shift(-1)
            #考虑5Tick趋势不变 >0  去除小突刺
        def Delete_5equal(x):
            if (x==1).sum() < 2:
                return 0
            else:
                return 1
        self.Tick['Delete_5equal'] = self.Tick['is_transi'].rolling(window=5).\
                                                        apply(Delete_5equal)
        for i in self.Tick[self.Tick.Delete_5equal == 1].index:
            if (self.Tick.loc[i+1,'diff1_EMAmedian'] * self.Tick.loc[i-3,'diff1_EMAmedian'] >0) and\
            (self.Tick.loc[i-2:i,'diff1_EMAmedian'].cumprod().iloc[-1]==0):
                self.Tick.loc[i-4:i,'is_transi'] = 0
            #相邻间隔趋势不变，解决长平
        index_transi = self.Tick[self.Tick['is_transi'] == 1].index
        for i in range(index_transi.shape[0]-1):
            thisi = index_transi[i]
            nexti = index_transi[i+1]
            if (sum(self.Tick.loc[thisi+1:nexti,'diff1_EMAmedian']) == 0)\
            and (self.Tick.loc[thisi,'diff1_EMAmedian'] * self.Tick.loc[nexti+1,'diff1_EMAmedian']>0):
                self.Tick.loc[thisi,'is_transi'] = 0
                self.Tick.loc[nexti,'is_transi'] = 0
        
        # 所有可能为转变点的index (包含振荡点)
        index_transi = self.Tick[self.Tick['is_transi'] == 1].index
        
        # 统计是否为趋势 30Tick周期以上的为趋势部分 其余为非趋势部分
        # index_transi 是统计的每个趋势的划分点，我们要做的就是计算index_transi相邻间隔
        TrendPeriod = 15
        trendindex = pd.np.where(pd.Series(index_transi).diff() > TrendPeriod)[0]
        trend = pd.Series([0] * self.Tick.shape[0])
        for i in trendindex:
            thisindex = index_transi[i]
            lastindex = index_transi[i-1]
            trend[lastindex:thisindex+1] = 1
        
        # 趋势内的转变点 50%
        trenddataindex = trend[trend == 1].index
        TrendIndexList = list(pd.Series(trenddataindex).sample(2400,replace=True))
        # 转变点前后各3个 30%
        TransIndexList = []
        for i in trendindex:
            thisindex = index_transi[i]
            lastindex = index_transi[i-1]
            TransIndexList += range(thisindex-3,thisindex+4)
            TransIndexList += range(lastindex-3,lastindex+4)
        # 震荡点 20%  -- 减去趋势点后剩下的部分随便采样
        OsiIndex = [i for i in range(4730) if i not in trenddataindex]
        OscilIndexList = list(pd.Series(OsiIndex).sample(1000,replace=True))

        SampleIndex = TrendIndexList + TransIndexList + OscilIndexList
        
        self.Tick = self.Tick.loc[SampleIndex].reset_index(drop=True)
        self.Tick = self.Tick.drop(['diff1_EMAmedian','former1',\
                                    'is_transi','Delete_5equal'],axis=1)
        return self.Tick















