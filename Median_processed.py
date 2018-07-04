# -*- coding: utf-8 -*-



import pandas as pd

# 600519  &  000725  & 000651
stock = "600519"

TickFile = "../data/TickData" + stock
TranFile = "../data/TransactionData" + stock

Tick0 = pd.HDFStore(TickFile,'r')           #读入Tick数据
Tran0 = pd.HDFStore(TranFile,'r')           #读入Tran数据

date = Tick0.keys()

tempTick = Tick0[date[-3]]


def smooth(temp):
    '''中位数平滑后，用此方法去除剩余的3tick不平滑点
        temp 是median平滑后的数据'''
    abnormal_index = []
    for i in range(1,len(temp) - 1):
        thisdiff = temp[i] - temp[i-1]
        nextdiff = temp[i+1] - temp[i]
        if  thisdiff * nextdiff < 0:
            abnormal_index.append(i)
            temp[i]   -= thisdiff
    print ("totally %s abnormal points fixed!"%(len(abnormal_index)))
    return temp,abnormal_index


AskP0 = tempTick.AskP0
median_5 = (tempTick['Price'].rolling(window=5).median()).shift(-2)
median_5_fixed,_ = smooth(median_5)
tempTick['median_5'] = median_5_fixed
median_9,_ = smooth((tempTick['AskP0'].rolling(window=9).median()).shift(-4))

from matplotlib import pyplot as plt
plt.figure(1)
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.plot(AskP0.values,'r',label='AskP0')
plt.plot(median_5.values,'g',label='median_5')
plt.plot(median_5_fixed.values,'b',label='median_5_fixed')
plt.plot(median_9.values,'k',label='median_9_fixed')
plt.legend(fontsize=40)
plt.title('贵州茅台',fontsize=40)

diff_median_5 = median_5.diff()
for i in range(diff_median_5.shape[0]-1):
    if diff_median_5[i] * diff_median_5[i+1] < 0:
        print (i)
    


# 买卖10价研究
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
plt.plot(tempTick.median_5,c='g',linewidth=7)

# 买卖10价对价的曲线
tempTick['sumAskV'] = 0
tempTick['sumBidV'] = 0
for i in range(10):
    tempTick['sumAskV'] += tempTick['AskV' + str(i)]
    tempTick['sumBidV'] += tempTick['BidV' + str(i)]

Name = lambda x,y:x+str(y)
NameList = lambda x,y: [Name(x,i) for i in y]

AskSumName = NameList('cumSumAskV',range(10))
BidSumName = NameList('cumSumBidV',range(10))
AskName = NameList('AskV',range(10))
BidName = NameList('BidV',range(10))

tempTick[AskSumName] = tempTick[AskName].apply(pd.np.cumsum,axis=1)
tempTick[BidSumName] = tempTick[BidName].apply(pd.np.cumsum,axis=1)
plt.figure(3)
def findequal(hands):
    tempTick['Askindex'+str(hands)] = tempTick[AskSumName].apply\
                    (lambda x:x[x<1e2 * hands].shape[0],axis=1)
    tempTick['Bidindex'+str(hands)] = tempTick[BidSumName].apply\
                    (lambda x:x[x<1e2 * hands].shape[0],axis=1) 
    tempTick['AskPriceV'+str(hands)] = tempTick.apply\
        (lambda x:x['AskP'+str(int(pd.np.clip(x['Askindex'+str(hands)],0,9)))],axis=1)
    tempTick['BidPriceV'+str(hands)] = tempTick.apply\
        (lambda x:x['BidP'+str(int(pd.np.clip(x['Bidindex'+str(hands)],0,9)))],axis=1)
    plt.plot(tempTick['AskPriceV'+str(hands)],c='r',linewidth=1)
    plt.plot(tempTick['BidPriceV'+str(hands)],c='c',linewidth=1)

findequal(5)
findequal(10)
findequal(15)
findequal(20)


plt.twinx()
plt.plot(tempTick.Turnover + 0.5 * 1e7)
plt.ylim([0,1.9*1e7])





