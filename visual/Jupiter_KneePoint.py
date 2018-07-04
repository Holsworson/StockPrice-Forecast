# -*- coding: utf-8 -*-

'''
part 1
'''
# data prepare
import glob,re,os
from matplotlib import pyplot as plt
import pandas as pd
import time
import TranFeature,Tools_kjy
import warnings

warnings.filterwarnings('ignore')
pd.options.display.width = 200

os.chdir('C:/kongjy_special/实习工作内容总结/201805-201808华泰证券/Task4/')
# 文件名
TickNameList = glob.glob('Java_log_orders_600519/Tick*.csv')
TranNameList = glob.glob('Java_log_orders_600519/Tran*.csv')
# TestName是 开平仓点 的csv，需要和Tick以及Tran保持时间对齐
DateList = [re.search('\d{8}',i)[0] for i in TickNameList if re.search('\d{8}',i)]  
TestNameFun = lambda x: "Java_log_orders_600519\\600519.SH_back_test_info_" + x +".csv"
TestNameList = [TestNameFun(i) for i in DateList]
print (TickNameList[0])
print (TranNameList[0])
print (TestNameList[0])

'''
part 1
'''
# 数据读入 和 预处理
date_index = 4
date = re.search('\d{8}',TickNameList[date_index])[0]
tempTick = pd.read_csv(TickNameList[date_index])
tempTran = pd.read_csv(TranNameList[date_index])
tempTest = pd.read_csv(TestNameList[date_index])
# Tick数据处理
tempTick = tempTick.rename(columns={"Unnamed: 0":"Timestamp"})
tempTick['Timestamp'] = pd.to_datetime(tempTick['Timestamp'])
tempTick['median_5'] = (tempTick['AskP0'].rolling(window=5).median()).shift(-2).fillna(method='pad')
tempTick['median_5'],_ = Tools_kjy.smooth(tempTick['median_5'])
# 开平仓点提取
tickindex = tempTest[tempTest.order != " "].index
print ("Date : %s"%(date))
# Tran 数据处理
tempTran = tempTran.rename(columns={"Unnamed: 0":"Timestamp"})
tempTran['Timestamp'] = pd.to_datetime(tempTran['Timestamp'])
FeatureTran = TranFeature.TranFeature(tempTran,tempTick)
tempTick = FeatureTran._Tran_features()

#print (tempTick.iloc[:20,-11:])

'''
part 1
'''

# 分数判断函数 -- 全局 + 下单位点
# 输出全局的正确率 和 下单点的正确率
def OutScore(x, y):
    '''    x : 测试的单因子，    y : 待预测的量     '''
    Trueindex  = tempTick[tempTick.loc[:,x] == tempTick.loc[:,y]].index
    Trueindex_Order = Trueindex & tickindex
    TotalScore = Trueindex.shape[0] / tempTick.shape[0]
    OrderScore = Trueindex_Order.shape[0] / tickindex.shape[0]
    print("单因子%s，全局正确率:%.3f，下单点正确率:%.3f。"%(x,TotalScore,OrderScore))

def normal(periods):
    argName = 'PastTrend' + str(periods)
    tempTick[argName]  = tempTick['diff1'].rolling(window=periods,min_periods=1).sum().apply(Tools_kjy.trinary)
    OutScore(argName,'isadd_3Tick')

def weighted(periods):
    time_weights = [date_i * date_i for date_i in range(periods)]
    tempTick['PastTrend'+str(periods)+'_weight'] = tempTick['diff1'].rolling(window=periods).\
                                apply(lambda x:pd.np.multiply(x,time_weights).sum()/sum(time_weights)).apply(Tools_kjy.trinary)
    OutScore('PastTrend'+str(periods)+'_weight','isadd_3Tick')

# 采用过去几个Tick价格的变化 预测 将来3Tick的价格变化
# y -- 3Tick后的价格变化
tempTick['isadd_3Tick'] = tempTick['median_5'].diff(3).shift(-3).fillna(0).apply(Tools_kjy.trinary)
tempTick['isadd_1Tick'] = tempTick['median_5'].diff(1).shift(-1).fillna(0).apply(Tools_kjy.trinary)

# x -- 过去价格单因子
tempTick['diff1'] = tempTick['median_5'].diff().fillna(0)
normal(2);normal(3);normal(4);normal(5);normal(6);normal(7);normal(8);normal(9);normal(10);normal(11)
weighted(2);weighted(3);weighted(4);weighted(5);weighted(6);weighted(7);weighted(8);weighted(9);weighted(10);weighted(11)


# Median5二阶差分研究 , 并研究二阶差分 在图片上的点
tempTick['isadd_next'] = tempTick['median_5'].diff().shift(-1).fillna(0).apply(Tools_kjy.trinary)
tempTick['KneePoint'] = tempTick['median_5'].diff().diff().fillna(0)
    #对下一个点的估计是一阶差分 + 二阶差分
tempTick['NextDiff'] = (tempTick['median_5'].diff().fillna(0) + tempTick['KneePoint']).apply(Tools_kjy.trinary)
Tools_kjy.scatter_classification(tempTick,'KneePoint','median_5')
plt.show()



