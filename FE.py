# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import time
import Mometum,BidAsk,TransVolu,Pressure,Emotion,Price,self_Features
from datetime import datetime,timedelta
import Tools_kjy


class FeatureEngeerning(object):

    def __init__(self,Tick,Tran):
        self.Tick = Tick
        self.Tran = Tran

        self._drop()
        self._findFirstMatch()           #找到self.MatchItem
        self._drop_Contunous()


    def _drop(self):
        '''
        转换每天的数据格式，drop无效特征
        '''
        # 删除撤单数据
        self.Tran = self.Tran[self.Tran['FunctionCode']!='C']
        # 改变时间格式
        self.Tick = self.Tick.reset_index().rename(columns={'index':'Timestamp'})
        self.Tran = self.Tran.reset_index().rename(columns={'index':'Timestamp'})
        self.Tick['Timestamp'] = pd.to_datetime(self.Tick['Timestamp'])
        self.Tran['Timestamp'] = pd.to_datetime(self.Tran['Timestamp'])

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
        需要优化：： 耗时6s
        '''
        time1 = time.time()
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
        time2 = time.time()
        print ("_drop_Contunous costs %.3f seconds!"%(time2-time1))


    def add_features(self):
        '''
        增加特征
        '''
        self._Tran_features()
        self.ToolsKJY()
        self.F6_Price()
        self.F1_Mometum()
        self.F2_BidAsk()
        self.F3_TransVolu()
        self.F4_Pressure()
        self.F5_Emotion()
        self.F8_SelfFeatures()

        return self.Tick

    def _Tran_features(self):
        '''
        增加Transaction 方向的特征
        '''
        temp = self.Tran[['Timestamp','Price']].groupby("Timestamp").agg\
                ({"Price":[np.min,np.max,np.median,np.size]})
        temp.columns = ['LowPrice','HighPrice','MedianPrice','CountPrice']
        
        temp['MedianBid'] = self.Tran.loc[self.Tran.BSFlags=="B",['Timestamp',\
                'Price']].groupby('Timestamp').median()
        temp['MedianAsk'] = self.Tran.loc[self.Tran.BSFlags=="S",['Timestamp',\
                'Price']].groupby('Timestamp').median()
        temp['CountBid'] = self.Tran.loc[self.Tran.BSFlags=="B",['Timestamp',\
                'Volume']].groupby('Timestamp').size()
        temp['CountAsk'] = self.Tran.loc[self.Tran.BSFlags=="S",['Timestamp',\
                'Volume']].groupby('Timestamp').size()
        
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
        return self.Tick


    def ToolsKJY(self):
        Tools_kjy.cumsum(self.Tick)
        Tools_kjy.findequal(self.Tick,5)
        Tools_kjy.findequal(self.Tick,10)
        Tools_kjy.findequal(self.Tick,15)
        Tools_kjy.findequal(self.Tick,20)
        print ('等体量价格因子done')
        Tools_kjy.BidAskPower(self.Tick,"Bid")
        Tools_kjy.BidAskPower(self.Tick,"Ask")
        self.Tick['TRAN_P1'] = (-self.Tick['BidStrength'] + self.Tick['AskStrength']).apply(Tools_kjy.trinary)
        # 盘口 买方 卖方强度因子 
        Tools_kjy.BidAskPower2(self.Tick,"Bid")
        Tools_kjy.BidAskPower2(self.Tick,"Ask")
        self.Tick['TRAN_P2'] = (-self.Tick['BidStrength2'] + self.Tick['AskStrength2']).apply(Tools_kjy.trinary)
        # 时间尺度上 等量 买方 卖方强度因子 
        periods = 6
        Tools_kjy.BidAskTime(self.Tick,"Bid",periods)
        Tools_kjy.BidAskTime(self.Tick,"Ask",periods)
        self.Tick['TRAN_P3'] = (-self.Tick['BidStrength'+str(periods)]  + self.Tick['AskStrength'+str(periods)]).apply(Tools_kjy.trinary)
        # 时间尺度上 盘口 买方 卖方强度因子 
        periods = 6
        Tools_kjy.BidAskTime2(self.Tick,"Bid",periods)
        Tools_kjy.BidAskTime2(self.Tick,"Ask",periods)
        self.Tick['TRAN_P4'] = (-self.Tick['BidStrength2'+str(periods)] + self.Tick['AskStrength2'+str(periods)]).apply(Tools_kjy.trinary)
        print ('ToolsKJY All done!')

    def F1_Mometum(self):
        '''
        动量型
        '''
        Mometum.PLUSDM(self.Tick)
        Mometum.MINUSDM(self.Tick)
        Mometum.TR(self.Tick)

        Mometum.PLUSDI(self.Tick)
        Mometum.PLUSDI(self.Tick,8)
        Mometum.PLUSDI(self.Tick,12)
        Mometum.PLUSDI(self.Tick,5)
        Mometum.PLUSDI(self.Tick,3)

        Mometum.MINUSDI(self.Tick)
        Mometum.MINUSDI(self.Tick,8)
        Mometum.MINUSDI(self.Tick,12)
        Mometum.MINUSDI(self.Tick,5)
        Mometum.MINUSDI(self.Tick,3)

        Mometum.DX(self.Tick)
        Mometum.DX(self.Tick,8)
        Mometum.DX(self.Tick,12)
        Mometum.DX(self.Tick,5)
        Mometum.DX(self.Tick,3)

        Mometum.ADX(self.Tick)
        Mometum.ADX(self.Tick,8)
        Mometum.ADX(self.Tick,12)
        Mometum.ADX(self.Tick,5)
        Mometum.ADX(self.Tick,3)

        Mometum.ADXR(self.Tick)
        Mometum.ADXR(self.Tick,8)
        Mometum.ADXR(self.Tick,12)
        Mometum.ADXR(self.Tick,5)
        Mometum.ADXR(self.Tick,3)

        Mometum.AMV(self.Tick)

        Mometum.APO(self.Tick)
        Mometum.APO(self.Tick,5,10)
        Mometum.APO(self.Tick,4,8)
        Mometum.APO(self.Tick,3,7)
        Mometum.APO(self.Tick,3,10)

        Mometum.ASI(self.Tick)

        Mometum.AROON(self.Tick)
        Mometum.AROON(self.Tick,20)
        Mometum.AROON(self.Tick,60)
        Mometum.AROON(self.Tick,120)
        Mometum.AROON(self.Tick,10)

        Mometum.AROONOSC(self.Tick)
        Mometum.AROONOSC(self.Tick,20)
        Mometum.AROONOSC(self.Tick,60)
        Mometum.AROONOSC(self.Tick,120)
        Mometum.AROONOSC(self.Tick,10)


        Mometum.BBI(self.Tick) # 均线价格相关暂时不想调整

        Mometum.BOP(self.Tick)

        Mometum.CMO(self.Tick)
        Mometum.CMO(self.Tick,20)
        Mometum.CMO(self.Tick,60)
        Mometum.CMO(self.Tick,120)
        Mometum.CMO(self.Tick,30)

        Mometum.DMA(self.Tick) # 均线相关暂时不想调整
        Mometum.ENV(self.Tick)
        Mometum.LWR(self.Tick)

        Mometum.MACD(self.Tick)
        Mometum.MACD(self.Tick,3,6,5)
        Mometum.MACD(self.Tick,4,8,7)
        Mometum.MACD(self.Tick,2,4,3)

        Mometum.TRIX(self.Tick)
        Mometum.TRIX(self.Tick,5)
        Mometum.TRIX(self.Tick,11)
        Mometum.TRIX(self.Tick,2)
        Mometum.TRIX(self.Tick,7)

        Mometum.MTM(self.Tick)
        Mometum.MTM(self.Tick,12)
        Mometum.MTM(self.Tick,4)
        Mometum.MTM(self.Tick,7)
        Mometum.MTM(self.Tick,9)

        Mometum.PBX(self.Tick)
        Mometum.PBX(self.Tick,5)
        Mometum.PBX(self.Tick,3)
        Mometum.PBX(self.Tick,15)
        Mometum.PBX(self.Tick,2)

        #Mometum.MATRIX(self.Tick)
        #Mometum.QACD(self.Tick)
        Mometum.ULTOSC(self.Tick)
        Mometum.UOS(self.Tick)

    def F2_BidAsk(self):
        '''
        超买超卖型
        '''
        BidAsk.ATR(self.Tick)

        BidAsk.BIAS(self.Tick)
        BidAsk.BIAS(self.Tick,7)
        BidAsk.BIAS(self.Tick,20)
        BidAsk.BIAS(self.Tick,4)
        BidAsk.BIAS(self.Tick,14)

        BidAsk.CCI(self.Tick)
        BidAsk.CCI(self.Tick,3)
        BidAsk.CCI(self.Tick,5)
        BidAsk.CCI(self.Tick,7)
        BidAsk.CCI(self.Tick,13)

        BidAsk.DPO(self.Tick)

        BidAsk.KDJ(self.Tick)
        BidAsk.KDJ(self.Tick,7)
        BidAsk.KDJ(self.Tick,15)
        BidAsk.KDJ(self.Tick,3)

        BidAsk.MFI(self.Tick)
        BidAsk.MFI(self.Tick,7)
        BidAsk.MFI(self.Tick,15)
        BidAsk.MFI(self.Tick,3)

        BidAsk.NATR(self.Tick)
        BidAsk.OSC(self.Tick)

        BidAsk.PPO(self.Tick)
        BidAsk.PPO(self.Tick,14,7)
        BidAsk.PPO(self.Tick,10,5)
        BidAsk.PPO(self.Tick,4,2)

        BidAsk.ROC(self.Tick)
        BidAsk.ROC(self.Tick,5)
        BidAsk.ROC(self.Tick,7)
        BidAsk.ROC(self.Tick,10)
        BidAsk.ROC(self.Tick,15)

        BidAsk.ROCP(self.Tick)

        BidAsk.ROCR(self.Tick)
        BidAsk.ROCR(self.Tick,5)
        BidAsk.ROCR(self.Tick,7)
        BidAsk.ROCR(self.Tick,10)
        BidAsk.ROCR(self.Tick,15)

        BidAsk.ROCR100(self.Tick)

        BidAsk.RSI(self.Tick)
        BidAsk.RSI(self.Tick,3)
        BidAsk.RSI(self.Tick,7)
        BidAsk.RSI(self.Tick,14)

        BidAsk.WilliamR(self.Tick)


    def F3_TransVolu(self):
        '''
        成交量
        '''
        TransVolu.ADOSC(self.Tick)
        TransVolu.ADVOL(self.Tick)
        TransVolu.NVI(self.Tick)
        TransVolu.OBV(self.Tick)
        TransVolu.PVI(self.Tick)
        TransVolu.PVT(self.Tick)
        TransVolu.VOSC(self.Tick)
        TransVolu.WVAD(self.Tick)

    def F4_Pressure(self):
        '''
        压力与支撑面
        '''
        Pressure.BBANDS(self.Tick)
        Pressure.SAR(self.Tick)
        Pressure.SAREXT(self.Tick)

    def F5_Emotion(self):
        '''
        情绪指标
        '''
        Emotion.ARBR(self.Tick)
        Emotion.ARBR(self.Tick,7)
        Emotion.ARBR(self.Tick,13)
        Emotion.ARBR(self.Tick,17)

        Emotion.CR(self.Tick)

        Emotion.PSY(self.Tick)
        Emotion.PSY(self.Tick,3)
        Emotion.PSY(self.Tick,7)
        Emotion.PSY(self.Tick,13)
        Emotion.PSY(self.Tick,17)

        Emotion.VR(self.Tick)
        Emotion.VR(self.Tick,3)
        Emotion.VR(self.Tick,7)
        Emotion.VR(self.Tick,13)
        Emotion.VR(self.Tick,17)

    def F6_Price(self):
        '''
        曲线交叉指标  和  交叉指标
        '''

        Price.AVGPRICE(self.Tick)
        Price.MEDPRICE(self.Tick)
        Price.MIDPOINT(self.Tick)
        Price.MIDPRICE(self.Tick)
        Price.TYPPRICE(self.Tick)
        Price.WCLPRICE(self.Tick)

        Price.DEMA(self.Tick)
        Price.EMA(self.Tick)

        Price.KAMA(self.Tick)
        Price.KAMA(self.Tick,5)
        Price.KAMA(self.Tick,8)
        Price.KAMA(self.Tick,15)
        Price.KAMA(self.Tick,20)

        Price.SMA(self.Tick)
        Price.SMA2(self.Tick)

        Price.STDDEV(self.Tick)
        Price.STDDEV(self.Tick,15)
        Price.STDDEV(self.Tick,4)
        Price.STDDEV(self.Tick,7)

        Price.T3(self.Tick)
        Price.T3(self.Tick,7)

        Price.TEMA(self.Tick)

        Price.TMA(self.Tick)
        Price.TMA(self.Tick,5)

        Price.TR(self.Tick)
        Price.TRMA(self.Tick)
        Price.VMA(self.Tick)
        Price.WMA(self.Tick)

    def F8_SelfFeatures(self):
        self_Features.SumV(self.Tick, begin_number=0, end_number=4)
        self_Features.SumV(self.Tick, begin_number=0, end_number=3)
        self_Features.SumV(self.Tick, begin_number=0, end_number=2)
        self_Features.SumV(self.Tick, begin_number=1, end_number=3)
        self_Features.SumV(self.Tick, begin_number=1, end_number=4)
        self_Features.SumV(self.Tick, begin_number=2, end_number=4)
