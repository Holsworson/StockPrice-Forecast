# -*- coding: utf-8 -*-

'''
先做分类问题，分类做的好做预测
现在做的问题是回归问题，考虑根据自己提取的特征
去回归计算3Tick后的价格变化的值的大小
'''

import time
t1 =time.time()
import re
import FE,visual
import pandas as pd
import numpy as np
import warnings
import Tools_kjy,Sample
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows',130)


# parameter setting
ndays = 20
test_day = 1
PredictLabel = 'AskP0'
shift = 3

# 初始化
Tick0 = pd.HDFStore('../data/TickData600519','r')           #读入Tick数据
Tran0 = pd.HDFStore('../data/TransactionData600519','r')    #读入Tran数据
date = Tick0.keys()                                     #日期List
cnt = 0                                                 #日期统计
day_cnt = []                                            #每个日期的数据
Tick0200 = pd.DataFrame()                               #特征数据

###############################################################################
# Step 1 ：特征构建
for i in date[-ndays-test_day-shift:-shift]:
    '''取出每天的数据，并且提取特征'''
    cnt += 1
    print ("Now At %s day !"%(cnt))
    tempTick = Tick0[i]                                 
    tempTran = Tran0[i]
    
    #1.1 特征工程 -- 进入FE.py
    Feature_Generate = FE.FeatureEngeerning(tempTick,tempTran)
    tempTick = Feature_Generate.add_features()
    
    #1.2 预测的y值  -- dy/y dy y/y
    def y(x):
        global tempTick
        tempTick['Past_diff'+str(x)] = tempTick[PredictLabel].diff(x)
    y(2);y(3);y(4);y(5)
#    tempTick['Future_diff1'] = tempTick['AskP0'].diff(1).shift(-1).apply(Tools_kjy._compa0)
    tempTick['median_5'] = (tempTick[PredictLabel].rolling(window=5).median()).shift(-2)
    tempTick['median_5'],_ = Tools_kjy.smooth(tempTick['median_5'])
    tempTick['Future_diff'] = tempTick[PredictLabel].diff(3).shift(-3)
    tempTick['isadd'] = tempTick['Future_diff'].apply(Tools_kjy._compa0)
    
    #1.3 采样 -- 随机采样 && 排序采样  如果不是train，是test区域。跳过
#    if cnt <= ndays:
    sample = Sample.Sample(tempTick)
    tempTick = sample.sample()
    tempTick = tempTick.fillna(0)
        #删除Tick内的无效特征 -- AskPn AskVn BidPn BidVn
    AB = lambda A,B : [str(i)+str(j) for i in A for j in B]
    PriceDropList = AB(['AskP','BidP','AskV','BidV'],[i for i in range(1,10)])
    PriceDropList += ['Time','MatchItem','Date','High','Low']
    PriceDropList += [_ for _ in tempTick.columns if 'cumsum' in _]
    PriceDropList += [_ for _ in tempTick.columns if ('Askindex' in _ ) or ('Bidindex' in _)]
    tempTick = tempTick.drop(PriceDropList,axis=1)
    #1.4 将处理完的数据汇入Tick0200
    Tick0200 = Tick0200.append(tempTick)
    day_cnt.append(Tick0200.shape[0])
    



# 处理缓存 
del tempTick
del tempTran
Tick0.close()
Tran0.close()

#1.5 数据后处理，测试训练分割，保留数据
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

TickTime = Tick0200['Timestamp'].values
Tick0200 = Tick0200.reset_index(drop=True).drop(['Timestamp'],axis=1)  #删除时间
Tick0200 = Tick0200.fillna(0)
columns = Tick0200.columns
Ycolumns = ['median_5','Future_diff','isadd']
Xcolumns = [i for i in columns if i not in Ycolumns]
#Xcolumns = ['Turnover','CountBid','CountAsk','HighPrice','LowPrice','AskPriceV5',
#            'AskPriceV20','BidPriceV5','BidPriceV20','Bidd01','Askd01','BidStrength',
#            'AskStrength','BidStrength6','AskStrength6','TRAN_P1','TRAN_P2',
#            'TRAN_P3','TRAN_P4','Past_diff4','Past_diff5']
X = Tick0200.loc[:,Xcolumns]
Y = Tick0200.loc[:,Ycolumns]

# 数据归一化
MMS = preprocessing.MinMaxScaler()
X = MMS.fit_transform(X)
# 测试训练分割
#X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,shuffle=False)
split_index = day_cnt[-test_day-1]
X_train = pd.DataFrame(X[:split_index,:],columns=Xcolumns)
Y_train = Y[:split_index]
X_test = pd.DataFrame(X[split_index:,:],columns=Xcolumns).reset_index(drop=True)
Y_test = Y[split_index:].reset_index(drop=True)
del X,Y

# 保留分析
X_test_Pastdiff4 = Tick0200.loc[split_index:,'Past_diff4']
X_train_Pastdiff4 = Tick0200.loc[:split_index-1,'Past_diff4']
#X_train_price = X_train.Price
#X_test_price = X_test.Price
#X_train_AskP0 = X_train[PredictLabel]
#X_test_AskP0 = X_test[PredictLabel]
#TrainTime = TickTime[:split_index]
#TestTime = TickTime[split_index:]

print ('\n$$$$$$$$$$$ Data preprocessing has finished! $$$$$$$$$$$\n')

###############################################################################

#step 2 :特征筛选 以及 特征组合
from sklearn.model_selection import \
                cross_val_score,ShuffleSplit,cross_validate,TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier

#2.1 初始化特征选择矩阵
Feature_select = pd.DataFrame(index = Xcolumns)

#2.2 Embedded
clf = RandomForestClassifier(n_estimators=2,max_depth=8,n_jobs=-1)
scores = []

for i in range(X_train.shape[1]):
    Embedd_X_Train = X_train.iloc[:,i:i+1].values
    Embedd_Y_Train = Y_train.isadd.values
    score = cross_val_score(
            estimator = clf, \
            X = Embedd_X_Train, \
            y = Embedd_Y_Train, \
            scoring = 'accuracy', \
            cv = 4)
    mean_score = round(np.mean(score), 3)
    scores.append(mean_score)
Feature_select['single'] = scores

#2.2 Person
corr = Tick0200.corr() 
Feature_select['Person'] = corr.loc[Xcolumns,'Future_diff'].apply(abs)

#2.3 Feature_importance 打分
import xgboost as xgb
XGBRegressor = xgb.XGBRegressor(
             objective='reg:linear',
             learning_rate = 0.1,
             n_estimators= 80,
             max_depth= 15,
             min_child_weight= 2,
             gamma=0.9,                        
             subsample=0.8,
             n_jobs= -1,
             base_score = 0
            )
print ("begin fitting ......")
XGBRegressor.fit(X_train.values, Y_train.Future_diff.values)
# Feature_importance_
Feature_importance_ini = pd.DataFrame(\
                    XGBRegressor.feature_importances_,\
                    index=Xcolumns,\
                    columns=['Feature_importance_ini'])
Feature_select['Feature_importance_ini'] = Feature_importance_ini.Feature_importance_ini

#2.4 totally importance
MMS_feature = preprocessing.MinMaxScaler()
Feature_select.iloc[:,:]= MMS_feature.fit_transform(Feature_select.values)
Feature_select['total'] =   0.30 * Feature_select['single'] + \
                            0.20 * Feature_select['Person'] + \
                            0.50 * Feature_select['Feature_importance_ini']
Feature_select = Feature_select.sort_values(by='total',ascending=False)

#2.5 同源特征仅保留一到二个
multiCol = ['PLUSDI','MINUSDI','DX','ADX','ADXR','APO','EMA','AROONUP','AROONDOWN',\
     'RUi','DUi','CMO','MACD','DIF','DEA','BIAS','RSV','KT','DT',\
     'JT','PMF','NMF','MFI','PPO','ROC','ROCR','RSI','BR','PSY','tempU',\
     'tempD','VR','kaufman','STDDEV','SMA','T3','TMA','Bidd','Askd']
for col in multiCol:
    MatchPattern =  re.compile(col + '\d+')
    subFeature_select = [i for i in Feature_select.index if re.match(MatchPattern,i)]
    # 求最大的total_score所对应的index
    MAXITEM = (Feature_select.loc[subFeature_select,'total']).nlargest(1).index[0]
    # 从中删除最大index，求剩下的person
    subFeature_select.remove(MAXITEM)
    # 剩下的Item中如果有最大的Person小于0.1
    smallestPerson = corr.loc[subFeature_select,MAXITEM].nsmallest(1)
    
    if not smallestPerson.empty and smallestPerson.values[0] < 0.25:
        MAXITEM_2 = smallestPerson.index[0]
        print ("Yes,Find 1,they are %s & %s !"%(MAXITEM_2,MAXITEM))
        subFeature_select.remove(MAXITEM_2)
    print (subFeature_select)
    Feature_select = Feature_select.drop(subFeature_select,axis=0)            

# 更新train test矩阵，更新特征
X_train = X_train.loc[:,Feature_select.index]
X_test = X_test.loc[:,Feature_select.index]
columns = list(Feature_select.index)

print ('\n$$$$$$$$$$$ Feature Engieering has finished! $$$$$$$$$$$\n')

##################################################
#step 3 : 特征可视化--------特征可视化

# 观看趋势
#visualization = visual.visualization(Tick0200)
# 单特征对结果的分化
#visualization.plot_scatter('MACDEMA2EMA43',-2)
# 但特征的不同类别的分化
#visualization.plot_hist(2,-2)
# Heatmap
#visualization.correlation_headmap()
# 双特征对结果的分化
#visualization.double_scatter(3,-100,-2)

##################################################

#step 4 : 模型部分---------模型部分
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import \
                cross_val_score,ShuffleSplit,cross_validate,TimeSeriesSplit
# 4.1 模型预处理
XGBRegressor = xgb.XGBRegressor(
             objective='reg:linear',
             learning_rate = 0.1,
             n_estimators= 80,
             max_depth= 15,
             min_child_weight= 2,
             gamma=0.9,                        
             subsample=0.8,
             n_jobs= -1,
             base_score = 0
            )
cv_split = ShuffleSplit(n_splits=4,test_size=0.25, train_size=0.7)
base_score = cross_validate(estimator=XGBRegressor,\
                            X=X_train,\
                            y=Y_train.Future_diff,\
                            cv=cv_split,\
                            scoring='r2')
print ("Before Tuning,Train score: %.3f"%(base_score['train_score'].mean()))
print ("Before Tuning,Test  score: %.3f"%(base_score['test_score'].mean()))
print ("Before Tuning,Test  std:   %.3f"%(base_score['test_score'].std()))
# 设置基础accuracy_score0, 用于以下的迭代
accuracy_score0 = base_score['test_score'].mean()
print ('*-' * 10)
# 开始预测
XGBRegressor.fit(X_train, Y_train.Future_diff)
# Train结果分析
print ("Base-Model predicting Training......")
predict_trains = XGBRegressor.predict(X_train)
print ("training r2_score is :",r2_score(Y_train.Future_diff,predict_trains))
# Test结果分析
print ("Base-Model predicting Testing......")
predict_test = XGBRegressor.predict(X_test)
print ('Baseline PastDiff   is %.5f'%(r2_score(Y_test.Future_diff,X_test_Pastdiff4/4)))
print ("Baseline [0]*n      is %.5f"%(r2_score(Y_test.Future_diff,[0]*len(predict_test))))
print ("testing r2_score    is %.5f:"%(r2_score(Y_test.Future_diff,predict_test)))

#from matplotlib import pyplot as plt
#predict = pd.DataFrame()
#predict['time'] = TickTime[split_index:]
#predict['predict'] = predict_test
#predict['median_5'] = Y_test['median_5']
#predict['isadd'] = Y_test['isadd']
#predict['istrue'] = (predict['predict']==predict['isadd']).map({True:1,False:0})
#predict['Turnover'] = X_test['Turnover']
#predict = predict.sort_values(by=['time']).reset_index(drop=True)
#predict['time'] = predict['time'].apply(str)
#trueindex = predict[predict['istrue'] == 1].index
#Falseindex = predict[predict['istrue'] == 0].index
#plt.figure()
#plt.scatter(trueindex,predict.loc[trueindex,'median_5'],c='r',linewidth=6)
#plt.scatter(Falseindex,predict.loc[Falseindex,'median_5'],c='g',linewidth=6)
#plt.plot(predict.median_5)


#4.2 模型调参
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV

# 调试特征
# sklearn 自带的特征选择太慢，不如自己选
#XgbClassifier_rfe = RFECV(XgbClassifier,step=0.2,scoring='accuracy',cv=cv_split)
#XgbClassifier_rfe.fit(X_train.values, Y_train.isadd.values)
#Features = X_train.columns[XgbClassifier_rfe.get_support()]
#X_train = X_train.loc[:,Features]
Feature_importance = pd.DataFrame(XGBRegressor.feature_importances_,\
                                index=columns,columns=['Feature_importance'])
Feature_importance = Feature_importance.sort_values(\
                                by='Feature_importance',ascending=False)
NewFeature = Feature_importance[Feature_importance.Feature_importance \
                            > 1.0/ pow(Feature_importance.shape[0],2)*10].index
X_train = X_train.loc[:,NewFeature]
X_test = X_test.loc[:,NewFeature]
accuracy_score0 -= 0.1
while True:
    score = 0
    # 迭代特征，设置截至条件为分数下降
    XGBRegressor = xgb.XGBRegressor(
                 objective='reg:linear',
                 learning_rate = 0.1,
                 n_estimators= 120,
                 max_depth= 15,
                 min_child_weight= 2,
                 gamma=0.9,                        
                 subsample=0.8,
                 n_jobs= -1,
                 base_score = 0
                )
    cv_split = ShuffleSplit(n_splits=4,test_size=0.35, train_size=0.6)
    for train_,test_ in cv_split.split(X_train):
        XGBRegressor.fit(X_train.loc[train_,NewFeature],\
                          Y_train.loc[train_,'Future_diff'])
        predict_test = XGBRegressor.predict(X_train.loc[test_,NewFeature])
        score += r2_score(Y_train.loc[test_,'Future_diff'],predict_test)
    score = score/4.0
    print ("迭代 Test r2 score: %.3f"%(score))
    accuracy_score1 = score
    
    if accuracy_score1 - accuracy_score0<-0.001:
        # 选特征，如果分数减小0.002且只减小了一个特征就停止选特征，否则减少一个特征
        # 否则删除1/len/7的重要性的特征
        if (accuracy_score1 - accuracy_score0<-0.002) or \
           (Feature_importance.shape[0] - NewFeature.shape[0])== 1:
            print (Feature_importance)
            break
        else:
            print ("Here!!!########################")
            print ("X_train 's shape is : ",X_train.shape)
            X_train = X_train[NewFeature]
            X_test = X_test[NewFeature]
            Feature_importance = pd.DataFrame(XGBRegressor.feature_importances_,\
                            index=NewFeature,columns=['Feature_importance'])
            Feature_importance = Feature_importance.sort_values(\
                            by='Feature_importance',ascending=False)
            NewFeature = Feature_importance[:-1].index
    else:
        #分数小于0.002，接受新特征，更新数据集
        X_train = X_train[NewFeature]
        X_test = X_test[NewFeature]
        print ("Here!!!***************************")
        Feature_importance = pd.DataFrame(XGBRegressor.feature_importances_,\
                            index=NewFeature,columns=['Feature_importance'])
        Feature_importance = Feature_importance.sort_values(\
                            by='Feature_importance',ascending=False)
        NewFeature = Feature_importance[Feature_importance.Feature_importance \
                            > 1/ Feature_importance.shape[0]/14].index
        if NewFeature.shape[0] == Feature_importance.shape[0]:
            print ("Minus one!")
            NewFeature = Feature_importance[:-1].index
    
    accuracy_score0 = accuracy_score1
    
# 调试参数
param_grid = {
        'max_depth':[12],
        'n_estimators':[120] }
tune_model = GridSearchCV(estimator = XGBRegressor,\
                          param_grid = param_grid,\
                          scoring = 'r2',\
                          n_jobs = 1,\
                          cv = cv_split)
# 开始训练
tune_model.fit(X_train, Y_train.Future_diff)

print ("After Tuning,Train score: %.3f"%(tune_model.cv_results_['mean_train_score'].mean()))
print ("After Tuning,Test  score: %.3f"%(tune_model.cv_results_['mean_test_score'].mean()))
print ("After Tuning,Test  std:   %.3f"%(tune_model.cv_results_['std_test_score'].std()))
print ('*-' * 10)
best_paras = tune_model.best_params_
# 开始预测
print ("Tune-Model predicting Training......")
predict_trains = tune_model.predict(X_train)
# Train结果分析
print ('Baseline PastDiff   is %.5f'%(r2_score(Y_train.Future_diff,X_train_Pastdiff4/4)))
print ("Baseline [0]*n      is %.5f"%(r2_score(Y_train.Future_diff,[0]*len(predict_trains))))
print ("Training r2_score   is %.5f:"%(r2_score(Y_train.Future_diff,predict_trains)))

# Test结果分析
print ("Tune-Model predicting Testing......")
predict_test = tune_model.predict(X_test)
accuracy_score0 = r2_score(Y_test.Future_diff,predict_test)
print ('Baseline PastDiff   is %.5f'%(r2_score(Y_test.Future_diff,X_test_Pastdiff4/4)))
print ("Baseline [0]*n      is %.5f"%(r2_score(Y_test.Future_diff,[0]*len(predict_test))))
print ("Testing r2 score    is %.5f"%(r2_score(Y_test.Future_diff,predict_test)))

print ('\n$$$$$$$$$$$ Modeling has finished! $$$$$$$$$$$\n')

'''
##################################################
#step 4 : 结果分析部分

#4.2 预测准确率分析
predict_proba = tune_model.predict_proba(X_test.values)
predict_proba = pd.DataFrame(predict_proba,columns=['0_prob','1_prob','2_prob'])
Y_test = pd.concat([Y_test,predict_proba],axis=1)
Y_test['prob'] = predict_proba.max(axis=1).values

#4.3 Y-test colume增加Price 和 AskP0
locPrice = Tick0200.columns.get_loc('Price')
Y_test['Price'] = X_test.loc[:,'Price'].values * \
                        MMS.data_range_[locPrice] + MMS.data_min_[locPrice]
#locAskP0 = Tick0200.columns.get_loc(PredictLabel)
#Y_test[PredictLabel] = X_test.loc[:,PredictLabel].values * \
#                        MMS.data_range_[locAskP0] + MMS.data_min_[locAskP0]
#判断结果的准确率
Y_test['predict_test'] = predict_test
A = (pd.Series(predict_test==Y_test['isadd'].values)).map({True:1,False:0})
Y_test.loc[:,'isRight'] = A.values

# 可视化
visual.evaluate(Y_test,predict_test)
#'可以选isadd & Future_diff1'
#visual.evaluate_2(Y_test,predict_test,locAskP0,'Future_diff')
#visual.evaluate_2(Y_test,predict_test,locAskP0,'isadd')
visual.PCA_visual(X_train,Y_train)

# 增长较大的区间分析
nls = 100.0
YnLargeIndex = Y_test.reset_index(drop=True).Future_diff.nlargest(int(nls)).index
YnSmallIndex = Y_test.reset_index(drop=True).Future_diff.nsmallest(int(nls)).index

Y_predict_Large = predict_test[YnLargeIndex]
Y_predict_Small = predict_test[YnSmallIndex]

print ('max descend <  0, True : %s !' %(sum(Y_predict_Small <  1)/nls))
print ('max ascend  >  0, True : %s !' %(sum(Y_predict_Large >  1)/nls))
print ('max descend <= 0, True : %s !' %(sum(Y_predict_Small <= 1)/nls))
print ('max ascend  >= 0, True : %s !' %(sum(Y_predict_Large >= 1)/nls))

occu_list = [] ; accu_list = []
for i in range(3,10):
    check_accu = 0.1 * i
    Y_temp = Y_test[(Y_test.prob>check_accu) & (Y_test.prob<check_accu+0.1)]
    try:
        occu_list.append(Y_temp.shape[0]/Y_test.shape[0])
        accu_list.append(Y_temp.isRight.sum()/Y_temp.shape[0])

        print ('预测区间(%.1f,%.1f)，占比为%.3f，置信水平为%.3f'%(check_accu,\
                       check_accu+0.1,occu_list[-1],accu_list[-1]))
    except ZeroDivisionError:
        pass

print ("\n$$$$$$$$$$$ All is done ! Good job! $$$$$$$$$$$\n")

# last part --- save the output
Y_test.index = TestTime
Y_test.to_csv('output_test.csv')
#读入文件
try:
    innn = pd.read_csv('Java_log_orders_600519/600519.SH_back_test_info_20180521.csv')
    tickindex = innn[innn.order != " "].index
except:
    pass

t2 = time.time()
print ("Total time : %s !"%(t2-t1))

'''