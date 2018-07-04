# -*- coding: utf-8 -*-

'''
单因子准确度评估
'''

import glob,re
from matplotlib import pyplot as plt
import pandas as pd
import time
import TranFeature
import warnings
warnings.filterwarnings('ignore')

# 600519  &  000725  & 000651
stock = "600519"
TickFile = "../data/TickData" + stock
TranFile = "../data/TransactionData" + stock
Tick0 = pd.HDFStore(TickFile,'r')           #读入Tick数据
Tran0 = pd.HDFStore(TranFile,'r')           #读入Tran数据

# 日期修正
date = Tick0.keys()
real_list = glob.glob('Java_log_orders_600519/*.csv')
# 开平仓的日期
date_real = [re.search('\d{8}',i)[0] for i in real_list]
# 开平仓 和 TickData日期的交集
date_now = [i for i in date if i[1:] in date_real]
real_list = [i for i in real_list if '/'+re.search('\d{8}',i)[0] in date_now ]

# 数据初始化
predict_data = {}
date_unique = []
tick_index = []


# functions
Name = lambda x,y:x+str(y)
NameList = lambda x,y: [Name(x,i) for i in y]

def binary(x):
    if (x['isadd'] > 1) and (x['predict_test'] < 1):
        return 0
    elif (x['isadd'] < 1) and (x['predict_test'] > 1):
        return 0
    else:
        return 1

def trinary(x):
    '''-1 0 1 映射到 0 1 2'''
    if   x < 0    :        return 0
    elif x > 0    :        return 2
    elif x==0 : return 1
#    else :raise BaseException ("存在None")
    
def findequal(tempTick,hands):
    
    tempTick['Askindex'+str(hands)] = tempTick[AskSumName].apply\
                    (lambda x:x[x<1e2 * hands].shape[0],axis=1)
    tempTick['Bidindex'+str(hands)] = tempTick[BidSumName].apply\
                    (lambda x:x[x<1e2 * hands].shape[0],axis=1) 
    tempTick['AskPriceV'+str(hands)] = tempTick.apply\
        (lambda x:x['AskP'+str(int(pd.np.clip(x['Askindex'+str(hands)],0,9)))],axis=1)
    tempTick['BidPriceV'+str(hands)] = tempTick.apply\
        (lambda x:x['BidP'+str(int(pd.np.clip(x['Bidindex'+str(hands)],0,9)))],axis=1)
#    plt.plot(tempTick['AskPriceV'+str(hands)],c='r',linewidth=1)
#    plt.plot(tempTick['BidPriceV'+str(hands)],c='c',linewidth=1)

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
    print ("totally %s abnormal points fixed!"%(len(abnormal_index)))
    return temp,abnormal_index

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
    
AskSumName = NameList('cumSumAskV',range(10))
BidSumName = NameList('cumSumBidV',range(10))
AskName = NameList('AskV',range(10))
BidName = NameList('BidV',range(10))

accuracy_PJ1 = 0
accuracy_PJ2 = 0
accuracy_PJ3 = 0
accuracy_PJ4 = 0
accuracy_PT6 = 0
accuracy_PT6_weight = 0
cnt = 0
for i in range(len(date_now))[-7:-6]:
    print ("日期:%s"%(date_now[i]))
    tempTick = Tick0[date_now[i]].reset_index().rename(columns={"index":"Timestamp"})
    tempTick['median_5'] = (tempTick['AskP0'].rolling(window=5).median()).shift(-2).fillna(method='pad')
    tempTick['median_5'],_ = smooth(tempTick['median_5'])
    tempTick[AskSumName] = tempTick[AskName].apply(pd.np.cumsum,axis=1)
    tempTick[BidSumName] = tempTick[BidName].apply(pd.np.cumsum,axis=1)
    
    tempTran = Tran0[date_now[i]].reset_index().rename(columns={"index":"Timestamp"})
    tempTran['Timestamp'] = pd.to_datetime(tempTran['Timestamp'])
    FeatureTran = TranFeature.TranFeature(tempTran,tempTick)
    tempTick = FeatureTran._Tran_features()
    
    time1 = time.time()
    findequal(tempTick,5)
    findequal(tempTick,10)
    findequal(tempTick,15)
    findequal(tempTick,20)
    findequal(tempTick,30)
    time2 = time.time()
    print ("Total cost %.3f seconds At findequal!"%(time2-time1))
    
    # real trible
    tempTick['diff3'] = tempTick['median_5'].diff(3).shift(-3).fillna(0)
    tempTick['isadd3'] = tempTick['diff3'].apply(trinary)
    tempTick['isadd_ori'] = tempTick['Price'].diff(1).shift(-1).fillna(0).apply(trinary)
    
    # predict trible -- single factors
        # 等量 买方 卖方强度因子
    BidAskPower(tempTick,"Bid")
    BidAskPower(tempTick,"Ask")
    tempTick['TRAN_P1_ori'] = (-tempTick['BidStrength'] + tempTick['AskStrength'])
    tempTick['TRAN_P1'] = tempTick['TRAN_P1_ori'].apply(trinary)
        # 盘口 买方 卖方强度因子 
    BidAskPower2(tempTick,"Bid")
    BidAskPower2(tempTick,"Ask")
    tempTick['TRAN_P2_ori'] = (-tempTick['BidStrength2'] + tempTick['AskStrength2'])
    tempTick['TRAN_P2'] = tempTick['TRAN_P2_ori'].apply(trinary)
        # 时间尺度上 等量 买方 卖方强度因子 
    periods = 6
    BidAskTime(tempTick,"Bid",periods)
    BidAskTime(tempTick,"Ask",periods)
    tempTick['TRAN_P3_ori'] = (-tempTick['BidStrength'+str(periods)] + tempTick['AskStrength'+str(periods)])
    tempTick['TRAN_P3'] = tempTick['TRAN_P3_ori'].apply(trinary)
        # 时间尺度上 盘口 买方 卖方强度因子 
    periods = 6
    BidAskTime2(tempTick,"Bid",periods)
    BidAskTime2(tempTick,"Ask",periods)
    tempTick['TRAN_P4_ori'] = (-tempTick['BidStrength2'+str(periods)] + tempTick['AskStrength2'+str(periods)])
    tempTick['TRAN_P4'] = tempTick['TRAN_P4_ori'].apply(trinary)
        # Transaction上的强度指标
        
    
#         卖一 和 买一之间的强度
#    time_weights = [pow(2,date_i) for date_i in range(periods)]
#    tempTick['Bid_P5'] = tempTick['BidP0'].diff().rolling(window=periods).apply(\
#            lambda x:pd.np.multiply(x,time_weights).sum())
#    tempTick['Ask_P5'] = tempTick['AskP0'].diff().rolling(window=periods).apply(\
#            lambda x:pd.np.multiply(x,time_weights).sum())
        # Past 6 trend
    tempTick['isadd01'] = tempTick['median_5'].diff().fillna(0)
    tempTick['PastTrend6'] = tempTick['isadd01'].rolling(window=6,\
                                        min_periods=1).sum().apply(trinary)
        # Past 6 Trend 加权
    periods = 4
    time_weights = [pow(2,date_i) for date_i in range(periods)]
    tempTick['PastTrend6_weight'] = tempTick['isadd01'].rolling(\
            window=periods,min_periods=periods).apply(\
            lambda x:pd.np.multiply(x,time_weights).sum()\
            /sum(time_weights)).apply(trinary)
    tempTick['PastTrend6_weight'] = tempTick['PastTrend6_weight'].fillna(0).apply(int)

    
    # open Java Log
    tempC = pd.read_csv(real_list[i])
    print ("open the JavaLog %s"%(real_list[i]))
    tickindex = tempC[tempC.order != " "].index 
    
    # get accuracy
    tempTick['isRight_P1'] = (tempTick['TRAN_P1'] == tempTick['isadd3']).map({True:1,False:0})
    accuracy_PJ1 += tempTick.loc[tickindex,'isRight_P1'].sum()\
                                /tempTick.loc[tickindex,'isRight_P1'].count()
    tempTick['isRight_P2'] = (tempTick['TRAN_P2'] == tempTick['isadd3']).map({True:1,False:0})
    accuracy_PJ2 += tempTick.loc[tickindex,'isRight_P2'].sum()\
                                /tempTick.loc[tickindex,'isRight_P2'].count()
    tempTick['isRight_P3'] = (tempTick['TRAN_P3'] == tempTick['isadd3']).map({True:1,False:0})
    accuracy_PJ3 += tempTick.loc[tickindex,'isRight_P3'].sum()\
                                /tempTick.loc[tickindex,'isRight_P3'].count()
    tempTick['isRight_P4'] = (tempTick['TRAN_P4'] == tempTick['isadd3']).map({True:1,False:0})
    accuracy_PJ4 += tempTick.loc[tickindex,'isRight_P4'].sum()\
                                /tempTick.loc[tickindex,'isRight_P4'].count()
    tempTick['PT_Right_6'] = (tempTick['PastTrend6'] == tempTick['isadd3']).map({True:1,False:0})
    accuracy_PT6 += tempTick.loc[tickindex,'PT_Right_6'].sum()/\
                                tempTick.loc[tickindex,'PT_Right_6'].count()
    tempTick['PT_Right_6_weight'] = (tempTick['PastTrend6_weight'] == tempTick['isadd3']).map({True:1,False:0})
    accuracy_PT6_weight += tempTick.loc[tickindex,'PT_Right_6_weight'].sum()/\
                                tempTick.loc[tickindex,'PT_Right_6_weight'].count()
    cnt += 1
    
print ("PJ 1 Accuracy : %.3f"%(accuracy_PJ1/cnt))
print ("PJ 2 Accuracy : %.3f"%(accuracy_PJ2/cnt))
print ("PJ 3 Accuracy : %.3f"%(accuracy_PJ3/cnt))
print ("PJ 4 Accuracy : %.3f"%(accuracy_PJ4/cnt))
print ("PT 6 Accuracy : %.3f"%(accuracy_PT6/cnt))
print ("PT 6 weight Accuracy : %.3f"%(accuracy_PT6_weight/cnt))    
    
    
    
    
PPlot = lambda x,y,c,l: plt.plot(tempTick[y+str(x)],c,linewidth=l)
plt.figure(2)
for i in range(10):
    if i == 9:
        c1 = c2 = 'g'
        l1 = l2 = 7
    else:
        c1,c2 = 'r','k'
        l1 = l2 = 1
    PPlot(i,"AskP",c1,l1)
    PPlot(i,"BidP",c2,l2)
plt.plot(tempTick.Price,c='b',linewidth=7)
plt.plot(tempTick.median_5[2:-2],c='g',linewidth=7)
plt.scatter(tickindex,tempTick.median_5[tickindex],c='r',marker='^',linewidths=20)

plt.figure(3)
for i in range(10):
    if i == 9:
        c1 = c2 = 'g'
        l1 = l2 = 7
    else:
        continue
        c1,c2 = 'r','k'
        l1 = l2 = 1
    PPlot(i,"AskP",c1,l1)
    PPlot(i,"BidP",c2,l2)
plt.plot(tempTick.Price,c='b',linewidth=7)
plt.plot(tempTick.median_5[2:-2],c='g',linewidth=7)

index1 = tickindex & tempTick[tempTick.PT_Right_6 == 1].index
index0 = tickindex & tempTick[tempTick.PT_Right_6 == 0].index
plt.scatter(index1 ,tempTick.median_5[index1],c='r',marker='^',linewidths=20)
plt.scatter(index0 ,tempTick.median_5[index0],c='k',marker='v',linewidths=20)
plt.plot(tempTick['HighPrice'],linewidth=5,c='k',label='HighPrice')
plt.plot(tempTick['MedianPrice'],linewidth=5,c='r',label='MedianPrice')
plt.plot(tempTick['LowPrice'],linewidth=5,c='k',label='LowPrice')
plt.plot(tempTick['MedianAsk'],linewidth=5,c='y',label='MedianAsk')
plt.plot(tempTick['MedianBid'],linewidth=5,c='r',label='MedianBid')
    
def findequal(hands):
    plt.plot(tempTick['AskPriceV'+str(hands)],c='r',linewidth=1)
    plt.plot(tempTick['BidPriceV'+str(hands)],c='c',linewidth=1)

findequal(5)
findequal(10)
findequal(20)
findequal(30)
plt.xticks(fontsize=20)
#plt.legend()
plt.twinx()
plt.plot(tempTick.Turnover + 0.5 * 1e7,c='b')
#plt.ylim([0,1.9*1e7])

    
    
    
    
    
TT = tempTick[['Price','MedianAsk','MedianBid','isadd_ori','BidPriceV5',\
               'AskPriceV5','CountBid','CountAsk','BidP0','AskP0']]
TT[['CountBid','CountAsk']] = TT[['CountBid','CountAsk']].fillna(0)
TT.loc[1.185e3:1.185e3+30,:]
    

tempTick['VAR_4'] =tempTick[['HighPrice','LowPrice','MedianAsk','MedianBid','median_5']].apply(pd.np.nanstd,axis=1)
tempTick['VAR_4_mean'] = tempTick['VAR_4'].rolling(window=4,min_periods=1).mean().shift(4)
tempTick['VAR_4'] = tempTick['VAR_4'] - tempTick['VAR_4_mean']
# 去掉Bid Ask 较大的区域
tempTick['BidAskGap'] = tempTick['AskPriceV5'] - tempTick['BidPriceV5']
BBG = tempTick.BidAskGap.nsmallest(4000).iloc[-1]
mm = tempTick.VAR_4.nsmallest(4000).iloc[-1]
plt.figure(7)
index2 = (tempTick[:4728][(tempTick.loc[:4728,'VAR_4']>mm) & (tempTick.loc[:4728,'BidAskGap']<BBG)]).index
plt.scatter(index2,tempTick.loc[index2,'median_5'],c='r')
plt.plot(tempTick.median_5[2:-2])

def trinary_kk(x):
    if abs(x) < 0.05:
        return 1
    elif x > 0.05:
        return 2
    elif x < -0.05:
        return 0

tempTick['newMedian2'] = tempTick['median_5'].apply(trinary_kk)
plt.figure(7)    
plt.plot(tempTick.Price)
#tempTick['newMedian2_trend'] = tempTick['newMedian2'].rolling(window=3,)
tt = (-tempTick['BidStrength'] + tempTick['AskStrength'])
dayu0index = tt[tt>0].index
dengyu0index = tt[tt==0].index
xiaoyu0index = tt[tt<0].index
plt.scatter(dengyu0index,tempTick.Price[dengyu0index],label='o',c='g')
plt.scatter(xiaoyu0index,tempTick.Price[xiaoyu0index],label='v',c='b')
plt.scatter(dayu0index,tempTick.Price[dayu0index],label='^',c='r')




# label
tempTick['EMA_median'] = tempTick['median_5'].rolling(window=9,min_periods=1)\
        .mean().shift(-4)
tempTick['std_EMAmedian'] = tempTick['median_5'].diff().rolling(window=15,min_periods=1)\
        .std().shift(-7)
plt.figure(5)
plt.plot(tempTick.AskP0,'b',label='AskP0',linewidth=5)
plt.plot(tempTick.median_5[:-10],'r',label='median_5',linewidth=5)
plt.plot(tempTick.EMA_median[:-10],'g',label='SMA_median',linewidth=5)
osimax = tempTick.std_EMAmedian.nlargest(700).iloc[-1]
#osimax = 1
osiminindex = tempTick[:-10].loc[tempTick[:-10].std_EMAmedian > osimax,:].index
plt.scatter(osiminindex,tempTick.loc[osiminindex,'EMA_median'],c='k',linewidths=10)
plt.legend(fontsize=20)


# 趋势反转点
tempTick['diff1_EMAmedian'] = tempTick['EMA_median'].diff()
tempTick['former1'] = tempTick['diff1_EMAmedian'].rolling(window=2).apply\
                (lambda x:pd.np.cumprod(x)[-1]).shift(-1)
def threecom(x):
    if (x[0] == 0) and (x[1]==0) and (x[2]==0):
        return 0
    if (x[0]>=0) and (x[1]<= 0) and (x[2]>=0):
        return 1
    else:
        return 0
tempTick['is_transi'] = tempTick['former1'].rolling(window=3).apply(threecom).shift(-1)
    #考虑5Tick趋势不变 >0  去除小突刺
def Delete_5equal(x):
    if (x==1).sum() < 2:
        return 0
    else:
        return 1
tempTick['Delete_5equal'] = tempTick['is_transi'].rolling(window=5).apply(Delete_5equal)
for i in tempTick[tempTick.Delete_5equal == 1].index:
    if (tempTick.loc[i+1,'diff1_EMAmedian'] * tempTick.loc[i-3,'diff1_EMAmedian'] >0) and\
    (tempTick.loc[i-2:i,'diff1_EMAmedian'].cumprod().iloc[-1]==0):
        tempTick.loc[i-4:i,'is_transi'] = 0
    #相邻间隔趋势不变，解决长平
index_transi = tempTick[tempTick['is_transi'] == 1].index
for i in range(index_transi.shape[0]-1):
    thisi = index_transi[i]
    nexti = index_transi[i+1]
    if (sum(tempTick.loc[thisi+1:nexti,'diff1_EMAmedian']) == 0)\
    and (tempTick.loc[thisi,'diff1_EMAmedian'] * tempTick.loc[nexti+1,'diff1_EMAmedian']>0):
        tempTick.loc[thisi,'is_transi'] = 0
        tempTick.loc[nexti,'is_transi'] = 0
        
plt.figure(8)
index_transi = tempTick[tempTick['is_transi'] == 1].index
plt.scatter(index_transi,tempTick.loc[index_transi,'EMA_median'],c='y',linewidths=7)
plt.plot(tempTick.loc[:4720,'median_5'],c='g')
plt.plot(tempTick.loc[:4720,'EMA_median'])
plt.plot(tempTick.loc[:4720,'AskP0'],c='k')
plt.scatter(tickindex,tempTick.loc[tickindex,'AskP0'],c='c',linewidths=7)

Ai = tempTick.loc[tickindex,'TRAN_P1_ori'].nlargest(60).index                  
plt.scatter(Ai,tempTick.loc[Ai,'AskP0'],linewidth=5,c='r')
Adi = tempTick['TRAN_P1_ori'].diff(6)[tickindex].nlargest(60).index
plt.scatter(Adi,tempTick.loc[Adi,'AskP0'],linewidth=5,c='b')

tempTick.isadd3[Adi].value_counts()
tempTick.isadd3[Ai].value_counts()
tempTick.isadd_ori[Adi].value_counts()
tempTick.isadd_ori[Ai].value_counts()
                                    
plt.twinx()
plt.plot(tempTick.Turnover ,c='b')
plt.ylim([0,4*1e7])







tempTick[['Price','AskP0','AskPriceV5','AskV0','BidP0','BidPriceV5','BidV0',\
          'isadd3','isadd_ori','MedianBid','CountBid','MedianAsk','CountAsk',\
          'HighPrice','LowPrice']].head(40)



