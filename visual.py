# -*- coding: utf-8 -*-


import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class visualization(object):
    
    def __init__(self,Tick):
        self.Tick = Tick
        self.column_list = Tick.columns
    
    def plot_single(self,column_name):
        '''
        查看单线的hist 和 变化
        '''
        if type(column_name) == int:
            column_name = self.column_list[column_name]
        plt.subplot(2,1,1)
        plt.plot(self.Tick.loc[:,column_name])
        plt.xlabel("Time")
        plt.ylabel(column_name)
        plt.subplot(2,1,2)
        plt.hist(self.Tick.loc[:,column_name])
        plt.xlabel('Time')
        plt.ylabel(column_name)
    
        
    def plot_scatter(self,column_name,yname):
        '''
        column_name 是 需要查看的column
        yname是预测name
        scatter图：
            纵坐标是预测的价格的变化
            横坐标是column_name的变化
            显示三种情况，不变，增加，减小的对应的column范围
        '''
        
        if type(column_name) == int:
            column_name = self.column_list[column_name]
        if type(yname) == int:
            yname = self.column_list[yname]
            
        temp = self.Tick.loc[:,[column_name,yname]]
        temp['isdayu0'] = (self.Tick.loc[:,yname]).map(self._compa0)
        plt.figure()
        plt.scatter(temp.loc[temp['isdayu0'] > 0,column_name],\
                                temp.loc[temp['isdayu0'] > 0,yname],c=["r"]
                                ,label="up")
        plt.scatter(temp.loc[temp['isdayu0'] == 0,column_name],\
                                temp.loc[temp['isdayu0'] == 0,yname],c=["g"]
                                ,label="equal")
        plt.scatter(temp.loc[temp['isdayu0'] < 0,column_name],\
                                temp.loc[temp['isdayu0'] < 0,yname],c=["b"]
                                ,label="down")
        plt.title('%s - self.Tick_change'%(column_name))
        plt.xlabel("%s"%(column_name))
        plt.ylabel("self.Tick")
        plt.legend()
        plt.grid()
        
    def plot_hist(self,column_name,yname):
        '''
        column_name 是 需要查看的column
        yname 是预测name
        hist图：
            显示hist三种情况，不变，增加，减小的对应的column范围
        '''
        
        if type(column_name) == int:
            column_name = self.column_list[column_name]
        if type(yname) == int:
            yname = self.column_list[yname]
            
        temp = self.Tick.loc[:,[column_name,yname]]
        temp['isdayu0'] = (self.Tick.loc[:,yname]).map(self._compa0)
        plt.figure()
        plt.hist(x=[temp.loc[temp['isdayu0'] > 0,column_name],\
                    temp.loc[temp['isdayu0'] == 0,column_name],\
                    temp.loc[temp['isdayu0'] < 0,column_name]],
                    stacked=True,color = ['r','g','b'],\
                    label=["up","equal","down"])
        plt.title('%s - self.Tick_change'%(column_name))
        plt.xlabel("%s"%(column_name))
        plt.ylabel("self.Tick")
        plt.legend()
    
    def correlation_headmap(self):
        '''
        heatmap
        '''
        colormap = sns.palplot(sns.diverging_palette(240, 10, n=11))
        sns.heatmap(self.Tick.iloc[:,:-1].corr(),
                cmap = colormap,
                cbar=True,#whether to add a colorbar
                annot=True,#whether to write the data
                )
    
    def double_scatter(self, C1, C2, yname):
        '''
        read _in list
        x-y column_name_list
        '''
        column_name_list = [C1, C2]
        if type(column_name_list[0]) == int:
            column_name_list = self.column_list[column_name_list]
        if type(yname) == int:
            yname = self.column_list[yname]
            
        plt.figure()
        temp = self.Tick.loc[:,column_name_list]
        temp['isdayu0'] = (self.Tick.loc[:, yname]).map(self._compa0)
        plt.scatter(temp.loc[temp['isdayu0']>0,column_name_list[0]],\
                    temp.loc[temp['isdayu0']>0,column_name_list[1]],\
                    c='r',label='up')
        plt.scatter(temp.loc[temp['isdayu0']==0,column_name_list[0]],\
                    temp.loc[temp['isdayu0']==0,column_name_list[1]],\
                    c='g',label='equal')
        plt.scatter(temp.loc[temp['isdayu0']<0,column_name_list[0]],\
                    temp.loc[temp['isdayu0']<0,column_name_list[1]],\
                    c='b',label='down')    
        plt.title('%s - %s' %(column_name_list[0],column_name_list[1]))
        plt.xlabel('%s'%(column_name_list[0]))
        plt.ylabel('%s'%(column_name_list[1]))
        plt.legend()
    
    def _compa0(self,x):
        if x > 0 : return 1
        elif x< 0 : return -1
        else :return 0 
        
        
def evaluate(Y_test,predict_test):
    '''
    画出测试集中大于0和等于-小于0
    三种情况下的分布情况
    '''
    

    
    Y_test_reset = Y_test.reset_index(drop=True)
    Y_P = Y_test_reset[Y_test_reset.Future_diff >  1]\
                        .sort_values(by=['Future_diff'],ascending=False).index
    Y_N = Y_test_reset[Y_test_reset.Future_diff <  1]\
                        .sort_values(by=['Future_diff'],ascending=True ).index
    Y_0 = Y_test_reset[Y_test_reset.Future_diff == 1]\
                        .sort_values(by=['Future_diff'],ascending=True ).index
                        
    interval = 50
    Y_predict_100_P = []
    Y_predict_100_Pprice = []
    Y_predict_100_N = []
    Y_predict_100_Nprice = []
    Y_predict_100_0 = []
    for i in range(int(Y_P.shape[0]/interval) + 1):
        Y_predict_100_P.append((1.0*predict_test\
                        [Y_P[i*interval:(i+1)*interval]] > 1).sum()/interval)
        Y_predict_100_Pprice.append(Y_test_reset.loc\
                        [Y_P[i*interval:(i+1)*interval],'Future_diff'].mean())
    for i in range(int(Y_N.shape[0]/interval) + 1):
        Y_predict_100_N.append((1.0*predict_test\
                        [Y_N[i*interval:(i+1)*interval]] < 1).sum()/interval)
        Y_predict_100_Nprice.append(Y_test_reset.loc\
                        [Y_N[i*interval:(i+1)*interval],'Future_diff'].mean())
    
    for i in range(int(Y_0.shape[0]/interval) + 1):
        Y_predict_100_0.append((1.0*predict_test[Y_0[i*interval:\
                        (i+1)*interval]] ==1).sum()/interval)
    
    # 按照价格去画数据
    plt.figure(figsize=(9,7))
    len0 = len(Y_predict_100_0)
    plt.plot(Y_predict_100_Nprice,Y_predict_100_N,label=u'价格跌')
    plt.plot(Y_predict_100_Pprice,Y_predict_100_P,label=u'价格涨')
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    plt.legend(fontsize=20)
    plt.xlabel(u'价格变化',fontsize=20)
    plt.ylabel(u'预测准确率',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.axis(ylim=[0,1])
    plt.title(u'不同涨跌幅中，预测的准确率',fontsize=20)
    
    # 按照百分比画图 --- 不包含
    plt.figure(figsize=(9,7))
    len0 = len(Y_predict_100_0)
    plt.plot(-np.arange(len(Y_predict_100_N))-len0/2,Y_predict_100_N[::-1],\
                         label=u'价格跌')
    plt.plot( np.arange(len(Y_predict_100_P))+len0/2,Y_predict_100_P[::-1],\
                         label=u'价格涨')
    plt.plot( np.arange( -len0/2, len0/2 )   ,Y_predict_100_0[::-1],\
                         label=u'价格不变')
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    plt.legend(fontsize=20)
    plt.xlabel(u'涨幅的大小，%s个数据一个点'%(interval),fontsize=20)
    plt.ylabel(u'预测准确率',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.axis(ylim=[0,1])
    plt.title(u'不同涨跌幅中，预测的准确率',fontsize=20)
    
    
    
    interval = 50
    Y_predict_100_P = []
    Y_predict_100_N = []
    for i in range(int(Y_P.shape[0]/interval) + 1):
        Y_predict_100_P.append((1.0*predict_test[Y_P[i*interval:\
                        (i+1)*interval]] >= 1).sum()/interval)
    for i in range(int(Y_N.shape[0]/interval) + 1):
        Y_predict_100_N.append((1.0*predict_test[Y_N[i*interval:\
                        (i+1)*interval]] <= 1).sum()/interval)
            
    plt.figure(figsize=(9,7))
    plt.plot(-np.arange(len(Y_predict_100_N)),Y_predict_100_N[::-1],\
                         label=u'价格跌(包含不变)')
    plt.plot( np.arange(len(Y_predict_100_P)),Y_predict_100_P[::-1],\
                         label=u'价格涨(包含不变)')
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    plt.legend(fontsize=20)
    plt.xlabel(u'涨幅的大小，%s个数据一个点'%(interval),fontsize=20)
    plt.ylabel(u'预测准确率',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.axis(ylim=[0,1])
    plt.title(u'不同涨跌幅中，预测的准确率',fontsize=20)
    
def evaluate_2(Y_test,predict_test,locPrice,types,line='AskP0'):
    '''
    Y_test       ： 预测的真值
    predict_test ： 预测的值
    locPrie      ： price在column中的location
    '''
    nls = 100.0
    YnLargeIndex = Y_test.reset_index(drop=True).\
                        Future_diff.nlargest(int(nls)).index
    YnSmallIndex = Y_test.reset_index(drop=True).\
                        Future_diff.nsmallest(int(nls)).index
    
    Y_predict_Large = predict_test[YnLargeIndex]
    Y_predict_Small = predict_test[YnSmallIndex]
    # 价格变化曲线，预测正确标准在图片上
#    plt.figure(figsize=(9,7))
#    plt.rcParams['font.sans-serif']=['SimHei']
#    plt.rcParams['axes.unicode_minus']=False
#    Y_test['predict_test'] = predict_test
#    A = (pd.Series(predict_test==Y_test[types].values)).map({True:1,False:0})
#    Y_test.loc[:,'isRight'] = A.values
#    plt.plot(Y_test['Price'])
#    TrueIndex = Y_test[Y_test.isRight == 1].index
#    FalseIndex = Y_test[Y_test.isRight == 0].index
#    plt.scatter(FalseIndex,Y_test.loc[FalseIndex,'Price']\
#                ,c='g',label='DOWN',marker='v')
#    plt.scatter(TrueIndex,Y_test.loc[TrueIndex,'Price']\
#                ,c='r',label='UP',marker='^')
    
    plt.figure(figsize=(20,14))
#    plt.rcParams['font.sans-serif']=['SimHei']
#    plt.rcParams['axes.unicode_minus']=False
    # Up_Eq 表示预测为涨的情况下，真实为不变
    Up_Up = Y_test[(Y_test[types] == 2) & (Y_test.predict_test == 2)].index
    Up_Eq = Y_test[(Y_test[types] == 1) & (Y_test.predict_test == 2)].index
    Up_Do = Y_test[(Y_test[types] == 0) & (Y_test.predict_test == 2)].index

    Eq_UP = Y_test[(Y_test[types] == 2) & (Y_test.predict_test == 1)].index
    Eq_Eq = Y_test[(Y_test[types] == 1) & (Y_test.predict_test == 1)].index
    Eq_Do = Y_test[(Y_test[types] == 0) & (Y_test.predict_test == 1)].index  
    
    Do_UP = Y_test[(Y_test[types] == 2) & (Y_test.predict_test == 0)].index
    Do_Eq = Y_test[(Y_test[types] == 1) & (Y_test.predict_test == 0)].index
    Do_Do = Y_test[(Y_test[types] == 0) & (Y_test.predict_test == 0)].index
    
    plt.subplot(3,1,1)
    plt.plot(Y_test.index,Y_test[line],label=line)
    plt.scatter(Up_Up,Y_test.loc[Up_Up,line],c='r',\
                                    label='Predict:增长,Real:增长',marker='^')
    plt.scatter(Up_Eq,Y_test.loc[Up_Eq,line],c='g',\
                                    label='Predict:增长,Real:不变',marker='o')
    plt.scatter(Up_Do,Y_test.loc[Up_Do,line],c='g',\
                                    label='Predict:增长,Real:降低',marker='v')
    plt.legend(fontsize=15)
    plt.yticks(fontsize=20)
    plt.grid()
    plt.title(u'预测为 增长 的预测准确率',fontsize=20)

    plt.subplot(3,1,2)
    plt.plot(Y_test.index,Y_test[line],label=line)
    plt.scatter(Eq_UP,Y_test.loc[Eq_UP,line],c='g',\
                                    label='Predict:不变,Real:增长',marker='^')
    plt.scatter(Eq_Eq,Y_test.loc[Eq_Eq,line],c='r',\
                                    label='Predict:不变,Real:不变',marker='o')
    plt.scatter(Eq_Do,Y_test.loc[Eq_Do,line],c='g',\
                                    label='Predict:不变,Real:降低',marker='v')
    plt.legend(fontsize=15)
    plt.yticks(fontsize=20)
    plt.grid()
    plt.title(u'预测为 不变 的预测准确率',fontsize=20)
    
    plt.subplot(3,1,3)
    plt.plot(Y_test.index,Y_test[line],label=line)
    plt.scatter(Do_UP,Y_test.loc[Do_UP,line],c='g',\
                                    label='Predict:降低,Real:增长',marker='^')
    plt.scatter(Do_Eq,Y_test.loc[Do_Eq,line],c='g',\
                                    label='Predict:降低,Real:不变',marker='o')
    plt.scatter(Do_Do,Y_test.loc[Do_Do,line],c='r',\
                                    label='Predict:降低,Real:降低',marker='v')
    plt.legend(fontsize=15)
    plt.yticks(fontsize=20)
    plt.grid()
    plt.title(u'预测为 降低 的预测准确率',fontsize=20)
    
    
def PCA_visual(X_train,Y_train):
    '''
    读入 X_train  和 Y_train
        e.g. PCA_visual(X_train,Y_train)
    '''
    from sklearn.decomposition import PCA
    from mpl_toolkits.mplot3d import Axes3D

    # 2-D
    pca = PCA(n_components=2)
    x_r = pca.fit(X_train).transform(X_train)
    UP = Y_train[Y_train.isadd == 2].index.values
    EQ = Y_train[Y_train.isadd == 1].index.values
    DO = Y_train[Y_train.isadd == 0].index.values
    plt.figure()
    plt.scatter(x_r[UP,0],x_r[UP,1],c='r',label='上涨',marker='^')
#    plt.scatter(x_r[EQ,0],x_r[EQ,1],c='b',label='不变',marker='o')
    plt.scatter(x_r[DO,0],x_r[DO,1],c='g',label='下跌',marker='v')
    plt.legend()
    
    # 3-D
    fig = plt.figure()
    pca = PCA(n_components=3)
    x_r = pca.fit(X_train).transform(X_train)
    ax = Axes3D(fig, elev=-150, azim=110)
    ax.scatter( x_r[UP,0],x_r[UP,1],x_r[UP,2],c='r',marker='^')
#    ax.scatter( x_r[EQ,0],x_r[EQ,1],x_r[EQ,2],c='b',marker='o')
    ax.scatter( x_r[DO,0],x_r[DO,1],x_r[DO,2],c='g',marker='v')
    plt.legend()

        
    