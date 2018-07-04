# -*- coding: utf-8 -*-

'''
计算预测的开仓点
预测的涨跌的正确率
'''
import pandas as pd
import glob,re
from matplotlib import pyplot as plt

real_list = glob.glob('Java_log_orders_600519/*.csv')
predict_list = glob.glob('output_sum/*.csv')
predict_data = {}
date_unique = []
tick_index = []

def binary(x):
    if (x['isadd'] > 1) and (x['predict_test'] < 1):
        return 0
    elif (x['isadd'] < 1) and (x['predict_test'] > 1):
        return 0
    else:
        return 1

def compa0(x):
    '''-1 0 1 映射到 0 1 2'''
    if x< 0:        return 0
    elif x ==0:     return 1
    elif x > 0:     return 2
    else:           print(x);raise BaseException ("查找下错误")
        


for i in range(len(predict_list)):
    '''
    提取预测的list {date:data}
    date : str
    data : pandas.core.DataFrame
    '''
    predictID = predict_list[i]
    temp = pd.read_csv(predictID,index_col=0).\
                            reset_index().rename(columns={"index":"Timestamp"})
    temp['Timestamp'] = pd.to_datetime(temp['Timestamp'])
    temp['month'] = temp['Timestamp'].apply(lambda x:x.month)
    #由于没有部分有5的情况，所以这儿没考虑
    if set(temp.month) != {5}:
        continue
    temp['date'] = temp['Timestamp'].apply(lambda x:x.day)
    temp['isRight2'] = temp.apply(binary,axis=1)
    temp['Price'] = temp['Future_diff'].cumsum()
    #增加历史趋势 作为 Baseline，前 1 3 5 tick的涨跌
    temp['isadd01'] = temp.median_5.diff().fillna(0)
    temp['PastTrend1'] = temp.isadd01.rolling(window=1,min_periods=1).sum().apply(compa0)
    temp['PT_Right_1'] = (temp.PastTrend1 == temp.isadd).map({True:1,False:0})
    temp['PastTrend2'] = temp.isadd01.rolling(window=2,min_periods=1).sum().apply(compa0)
    temp['PT_Right_2'] = (temp.PastTrend2 == temp.isadd).map({True:1,False:0})
    temp['PastTrend3'] = temp.isadd01.rolling(window=3,min_periods=1).sum().apply(compa0)
    temp['PT_Right_3'] = (temp.PastTrend3 == temp.isadd).map({True:1,False:0})
    temp['PastTrend4'] = temp.isadd01.rolling(window=4,min_periods=1).sum().apply(compa0)
    temp['PT_Right_4'] = (temp.PastTrend4 == temp.isadd).map({True:1,False:0})
    temp['PastTrend5'] = temp.isadd01.rolling(window=5,min_periods=1).sum().apply(compa0)
    temp['PT_Right_5'] = (temp.PastTrend5 == temp.isadd).map({True:1,False:0})
    temp['PastTrend6'] = temp.isadd01.rolling(window=6,min_periods=1).sum().apply(compa0)
    temp['PT_Right_6'] = (temp.PastTrend6 == temp.isadd).map({True:1,False:0})
    temp['PastTrend7'] = temp.isadd01.rolling(window=7,min_periods=1).sum().apply(compa0)
    temp['PT_Right_7'] = (temp.PastTrend7 == temp.isadd).map({True:1,False:0})
    temp['PastTrend8'] = temp.isadd01.rolling(window=8,min_periods=1).sum().apply(compa0)
    temp['PT_Right_8'] = (temp.PastTrend8 == temp.isadd).map({True:1,False:0})
    temp['PastTrend9'] = temp.isadd01.rolling(window=9,min_periods=1).sum().apply(compa0)
    temp['PT_Right_9'] = (temp.PastTrend9 == temp.isadd).map({True:1,False:0})
    
    this_date = temp['date'].unique()
    date_unique += list(this_date)
    
    for j in this_date:
        predict_data[str(j)] = temp[temp['date']==j].reset_index(drop=True)

accuracy3_sum = 0
average_PT1 = 0
average_PT2 = 0
average_PT3 = 0
average_PT4 = 0
average_PT5 = 0
average_PT6 = 0
average_PT7 = 0
average_PT8 = 0
average_PT9 = 0
for i in range(len(real_list)):
    '''
    如果真实开仓的日期在date_unique中
    开始计算正确率
    '''
    realID = real_list[i]
    this_date = re.search('201805\d\d',realID).group()[-2:]
    this_date = str(int(this_date))
    if int(this_date) not in date_unique:
        continue
    
    #开始计算正确率
    print ("日期:5月%s日!"%(this_date))
#    temp = pd.read_csv('Java_log_orders_600519/600519.SH_back_test_info_20180521.csv')
    temp = pd.read_csv(realID)
    tickindex = temp[temp.order != " "].index 
    tick_index.append(tickindex)
    tempPredict = predict_data[this_date]
    isRight = tempPredict.loc[tickindex,'isRight']
    isRight2 = tempPredict.loc[tickindex,'isRight2']
    preValueCounts = tempPredict.isadd.value_counts()
    
    accuracy3 = isRight.sum()/isRight.shape[0]
    baseline3 = preValueCounts.max() / preValueCounts.sum()
    accuracy2 = isRight2.sum()/isRight.shape[0]
    Baseline2 = (preValueCounts.max() + preValueCounts.get(0)) / preValueCounts.sum()
    
    accuracy_PT1 = (tempPredict.loc[tickindex,'PT_Right_1'] > 0 ).sum()/tickindex.shape[0]
    accuracy_PT2 = (tempPredict.loc[tickindex,'PT_Right_2'] > 0 ).sum()/tickindex.shape[0]
    accuracy_PT3 = (tempPredict.loc[tickindex,'PT_Right_3'] > 0 ).sum()/tickindex.shape[0]
    accuracy_PT4 = (tempPredict.loc[tickindex,'PT_Right_4'] > 0 ).sum()/tickindex.shape[0]
    accuracy_PT5 = (tempPredict.loc[tickindex,'PT_Right_5'] > 0 ).sum()/tickindex.shape[0]
    accuracy_PT6 = (tempPredict.loc[tickindex,'PT_Right_6'] > 0 ).sum()/tickindex.shape[0]
    accuracy_PT7 = (tempPredict.loc[tickindex,'PT_Right_7'] > 0 ).sum()/tickindex.shape[0]
    accuracy_PT8 = (tempPredict.loc[tickindex,'PT_Right_8'] > 0 ).sum()/tickindex.shape[0]
    accuracy_PT9 = (tempPredict.loc[tickindex,'PT_Right_9'] > 0 ).sum()/tickindex.shape[0]
    
    accuracy3_sum += accuracy3  
    average_PT1 += accuracy_PT1
    average_PT2 += accuracy_PT2
    average_PT3 += accuracy_PT3
    average_PT4 += accuracy_PT4
    average_PT5 += accuracy_PT5
    average_PT6 += accuracy_PT6
    average_PT7 += accuracy_PT7
    average_PT8 += accuracy_PT8
    average_PT9 += accuracy_PT9
    
    print ('三分类准确率: %.3f，Baseline:%.3f，\n二分类准确率: %.3f，Baseline:%.3f。' 
           %(accuracy3,baseline3,accuracy2,Baseline2))
    print ('过去1Tick统计: %.3f，过去2Tick统计: %.3f，过去3Tick统计: %.3f;'
           %(accuracy_PT1,accuracy_PT2,accuracy_PT3))
    print ('过去4Tick统计: %.3f，过去5Tick统计: %.3f，过去6Tick统计: %.3f;'
           %(accuracy_PT4,accuracy_PT5,accuracy_PT6))
    print ('过去7Tick统计: %.3f，过去8Tick统计: %.3f，过去9Tick统计: %.3f;'
           %(accuracy_PT7,accuracy_PT8,accuracy_PT9))
#    print ('三分类准确率: %.3f，Baseline:%.3f' 
#           %(accuracy3,baseline3))


#手动跳下，还有22号没有
plt.figure()
if 22 in date_unique:
    date_unique.remove(22)
date_unique = sorted(date_unique)
cnt = 0
for i in range(len(date_unique)):
    
    tempdata = predict_data[str(date_unique[i])]
    temp = tempdata.Price
    tickindex = tick_index[i]
    plt.plot(pd.np.arange(cnt,cnt+4730),temp,'k')
    
    isRight = tempdata.loc[tickindex,'isRight']
    TrueIndex = isRight[isRight==1].index
    FalseIndex = isRight[isRight==0].index
    plt.scatter(pd.np.arange(cnt,cnt+4730)[FalseIndex],temp[FalseIndex],c='g')
    plt.scatter(pd.np.arange(cnt,cnt+4730)[TrueIndex],temp[TrueIndex],c='r')

    cnt += 4730
    plt.grid()

print ("The average accuracy is %.4f!"%(accuracy3_sum/len(date_unique)))
print ("过去1Tick is %.4f!"%(average_PT1/len(date_unique)))
print ("过去2Tick is %.4f!"%(average_PT2/len(date_unique)))
print ("过去3Tick is %.4f!"%(average_PT3/len(date_unique)))
print ("过去4Tick is %.4f!"%(average_PT4/len(date_unique)))
print ("过去5Tick is %.4f!"%(average_PT5/len(date_unique)))
print ("过去6Tick is %.4f!"%(average_PT6/len(date_unique)))
print ("过去7Tick is %.4f!"%(average_PT7/len(date_unique)))
print ("过去8Tick is %.4f!"%(average_PT8/len(date_unique)))
print ("过去9Tick is %.4f!"%(average_PT9/len(date_unique)))
'''
plt.figure()
temp1 = temp.rolling(window=11,min_periods=1).median().shift(-5)
temp2 = temp.rolling(window=15,min_periods=1).median().shift(-7)
temp3 = temp.rolling(window=5,min_periods=1).mean().shift(-2)
temp3 = temp3.rolling(window=7,min_periods=1).median().shift(-3)
temp4 = temp.rolling(window=5,min_periods=1).mean().shift(-2)
temp5 = temp4.rolling(window=3,min_periods=1).mean().shift(-1)
temp6 = temp5.rolling(window=3,min_periods=1).mean().shift(-1)
temp7 = temp.rolling(window=5,min_periods=1).median().shift(-2)
temp8 = temp.rolling(window=7,min_periods=1).median().shift(-3)
temp9 = temp.rolling(window=3,min_periods=1).median().shift(-1)
plt.plot(temp,label='Original',linewidth=5)
#plt.plot(temp1,label='M1')
#plt.plot(temp2,label='M2')
#plt.plot(temp3,label='M3')
plt.plot(temp4,label='M4')
plt.plot(temp5,label='M5')
plt.plot(temp6,label='M6')
plt.plot(temp7,label='M7',linewidth=4)
plt.plot(temp8,label='M8',linewidth=4)
plt.plot(temp9,label='M9',linewidth=4)
plt.legend()
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)


temp_3tick = temp.rolling(window=3,min_periods=1).median().shift(-1)
temp_3tick_diff = temp_3tick.diff()
temp_3tick_diff = temp_3tick_diff.fillna(0)
temp_3tick_change = pd.Series([0.0]*len(temp_3tick))

abnormal_index = []
for i in range(len(temp_3tick) - 2):
    thisdiff = temp_3tick_diff[i]
    nextdiff = temp_3tick_diff[i+1]
    if  thisdiff * nextdiff < 0:
        print ('Here')
        abnormal_index.append(i)
        temp_3tick_change[i]   -= thisdiff

temp_3tick += temp_3tick_change
plt.plot(temp_3tick,label='MM',linewidth=5)     
plt.legend()

'''



