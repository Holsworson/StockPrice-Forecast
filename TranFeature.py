# -*- coding: utf-8 -*-

import pandas as pd
from datetime import datetime,timedelta
import time
import numpy as np

class TranFeature(object):
    
    def __init__(self,Tran,Tick):
        self.Tran = Tran
        self.Tick = Tick
        
        self._drop()
        self._findFirstMatch()           #找到self.MatchItem
        self._drop_Contunous()
        
    def _drop(self):
        '''
        转换每天的数据格式，drop无效特征
        '''

        if "MatchItem" not in self.Tick.columns:
            self._addMatchItem()
        # 使得MatchItem为整形
        self.Tick['MatchItem'] = self.Tick['MatchItem'].apply(int)


    def _addMatchItem(self):
        '''
        一天之内 Tick的时间段是：
        上午9.30.12 - 11.29.57
        下午13.00.15 - 14.56.57
        '''
        cnt = 0
        MatchItemList = []
        self.timeList = []
        _toDay = self.Tran.Timestamp[0].date()
        Morning_lag = datetime(_toDay.year,_toDay.month,_toDay.day,9,30,15)
        Morning_end = datetime(_toDay.year,_toDay.month,_toDay.day,11,29,57)
        Afternoon_lag = datetime(_toDay.year,_toDay.month,_toDay.day,13,00,15)
        Afternoon_end = datetime(_toDay.year,_toDay.month,_toDay.day,14,56,57)

        for i in self.Tran.index[:-2]:

            temptime = self.Tran.loc[i,'Timestamp']
            # 上午 9.30.12 - 11.29.57
            if (temptime.hour <= 11) :
                #中间点缺失数据，用while补齐
                while temptime > Morning_lag:
                    MatchItemList.append(cnt)
                    self.timeList.append(temptime)
                    Morning_lag += timedelta(seconds=3)
                cnt += 1

            elif (temptime.hour > 11) and (Morning_lag <= Morning_end):
                #11.29.50 - 57没有数据，需要补齐
                while (Morning_lag <= Morning_end):
                    MatchItemList.append(cnt)
                    Morning_lag += timedelta(seconds=3)

            elif temptime <= Afternoon_end + timedelta(seconds=3):

                while temptime > Afternoon_lag:
                    MatchItemList.append(cnt)
                    self.timeList.append(temptime)
                    Afternoon_lag += timedelta(seconds=3)
                cnt += 1

            elif (temptime.hour >= 14) and (Afternoon_lag <= Afternoon_end):
                #14.56.50-57 没有数据需要补齐
                while (Afternoon_lag <= Afternoon_end):
                    MatchItemList.append(cnt)
                    Afternoon_lag += timedelta(seconds=3)

            else:
                break
#        print (len(MatchItemList))
#        print (self.Tick.shape)
        self.Tick['MatchItem'] = MatchItemList


    def _findFirstMatch(self):
        '''
        找到Tick中没有的第一个MatchItem，
        创建MatchItem list
        '''
        for i in range(len(self.Tran)):
            _time = self.Tran.loc[i,'Timestamp']
            if _time.hour == 9 and _time.minute <= 30 and _time.second < 12:
                continue
            else:
                index0 = i
                break
        self.MatchItem = [index0] + list(self.Tick['MatchItem'])


    def _drop_Contunous(self):
        '''
        删掉Tran中的非Tick中时间段
        共享时间段，重新标注Tran中的时间段
        '''
        index0 = self.MatchItem[0]
        _Timestamp = list(self.Tran['Timestamp'])
        TickTime = list(self.Tick['Timestamp'])

        for i in range(len(self.MatchItem)-1):
            index1 = self.MatchItem[i]
            index2 = self.MatchItem[i+1]
            _Timestamp[index1:index2] = [TickTime[i]] * (index2-index1)
        self.Tran['Timestamp'] = _Timestamp
        self.Tran = self.Tran.drop(self.Tran.index[:index0]).\
                                            drop(self.Tran.index[index2:])


    def _Tran_features(self):
        '''
        增加Transaction 方向的特征
        '''
        temp = self.Tran[['Timestamp','Price']].groupby("Timestamp").agg\
                ({"Price":[np.min,np.max,np.median,np.size]})
        temp.columns = ['LowPrice','HighPrice','MedianPrice','CountPrice']
        
#        temp['MedianBid'] = self.Tran.loc[self.Tran.BSFlags=="B",['Timestamp',\
#                'Price']].groupby('Timestamp').median()
#        temp['MedianAsk'] = self.Tran.loc[self.Tran.BSFlags=="S",['Timestamp',\
#                'Price']].groupby('Timestamp').median()
#        temp['CountBid'] = self.Tran.loc[self.Tran.BSFlags=="B",['Timestamp',\
#                'Volume']].groupby('Timestamp').size()
#        temp['CountAsk'] = self.Tran.loc[self.Tran.BSFlags=="S",['Timestamp',\
#                'Volume']].groupby('Timestamp').size()

        temp['CountBid'] = self.Tran.loc[self.Tran.BSFlags=="B",['Timestamp',\
                'Volume']].groupby('Timestamp').sum()
        temp['CountAsk'] = self.Tran.loc[self.Tran.BSFlags=="S",['Timestamp',\
                'Volume']].groupby('Timestamp').sum()
        def MedianV_price(x):
            _ = []
            for i in x.index:
                _ += [x.loc[i,'Price']] * int(x.loc[i,'Volume']/100)
            return pd.Series(_).median()
        temp['MedianBid'] = self.Tran.loc[self.Tran.BSFlags=="B",\
            ['Timestamp','Price','Volume']].groupby('Timestamp').apply(MedianV_price)
        temp['MedianAsk'] = self.Tran.loc[self.Tran.BSFlags=="S",\
            ['Timestamp','Price','Volume']].groupby('Timestamp').apply(MedianV_price)
        
        # temp的index 是Timestamp --> 提出Timestamp 和 Tick merge
        temp = temp.reset_index().rename(columns={'index':'Timestamp'})
        self.Tick = pd.merge(self.Tick,temp,on=['Timestamp'],how='left')
        
        nullindex = self.Tick.HighPrice.isnull()
        self.Tick.loc[nullindex,'HighPrice'] = self.Tick.loc[nullindex,'Price']
        self.Tick.loc[nullindex,'LowPrice'] = self.Tick.loc[nullindex,'Price']
        self.Tick.loc[nullindex,'MedianPrice'] = self.Tick.loc[nullindex,'Price']
        nullBid = self.Tick['MedianBid'].isnull()
        self.Tick.loc[nullBid,'MedianBid'] = self.Tick.loc[nullBid,'Price']
        nullAsk = self.Tick['MedianAsk'].isnull()
        self.Tick.loc[nullAsk,'MedianAsk'] = self.Tick.loc[nullAsk,'Price']
        self.Tick['CountBid'] = self.Tick['CountBid'].fillna(0)
        self.Tick['CountAsk'] = self.Tick['CountAsk'].fillna(0)
        self.Tick['CountBid'] = self.Tick['CountBid'].apply(lambda x:int(x/100))
        self.Tick['CountAsk'] = self.Tick['CountAsk'].apply(lambda x:int(x/100))
        
        return self.Tick
    
    def _Tran_energy(self):
        '''
        TranEnergy = median(Tran) * entrophy(Tran)
        entrophy(Tran) = pi * ln(pi)
        pi = sorted by same interval's Price
            maxmin in the same interval and divede in the same length
        '''
        t1 = time.time()
        def entropy(lista):
            if not lista or sum([np.isnan(i) for i in lista])>0:
                return 0
            # minmax part
            clip_up = 2; clip_down = -2
            _ = [(i-clip_down)/(clip_up - clip_down) for i in lista ]
            # 将范围限制在0-1
            _ = _ + [0,1]
            # get the result
            _ = pd.Series(lista).value_counts(bins=400,normalize=True)
            entrophy = sum([-i*pd.np.log(i) for i in _ if i!=0])
            return entrophy
    
        def pp(x):
            _ = []
            for i in x.index:
                _ += [x.loc[i,'ChangePrice']] * int(x.loc[i,'Volume']/100)
            Entrophy = entropy(_)
            return Entrophy
        
        def MedianV_price(x):
            _ = []
            for i in x.index:
                _ += [x.loc[i,'ChangePrice']] * int(x.loc[i,'Volume']/100)
            return pd.Series(_).median()
        
        # 额外用到些数据 
        self.Tick['PriceShift1'] = self.Tick['Price'].shift(1)  # 用于计算能量
        
        self.Tran = pd.merge(self.Tran,self.Tick[['Timestamp','PriceShift1']],on='Timestamp',how='left')
        self.Tran['ChangePrice'] = (self.Tran['Price'] - self.Tran['PriceShift1']).fillna(0)
        temp = pd.DataFrame(self.Tran[['Timestamp','ChangePrice','Volume']].groupby('Timestamp').apply(pp))
        temp['PriceTempera'] = self.Tran[['Timestamp','ChangePrice','Volume']].\
                                    groupby('Timestamp').apply(MedianV_price)
        temp = temp.reset_index().rename(columns={'index':'Timestamp',0:'TickEntrophy'})
        temp['Timestamp'] = pd.to_datetime(temp['Timestamp'])
        
        self.Tick = pd.merge(temp,self.Tick,on=['Timestamp'],how='right')
        self.Tick['TickEntrophy'] = self.Tick['TickEntrophy'].fillna(0)
        self.Tick['PriceTempera'] = self.Tick['PriceTempera'].fillna(0)
        t2 = time.time()
        print ("_Tran_energy costs %.4f seconds."%(t2-t1))
        return self.Tick