# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt

def _compa0(x):
    if x > 0 : return 2
    elif x< 0 : return 0
    else :return 1

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
#    print ("totally %s abnormal points fixed!"%(len(abnormal_index)))
    return temp,abnormal_index

def scatter_classification(data,column_name,yname):
    temp = data.loc[:,[column_name,yname]]
    temp['isdayu0'] = (data.loc[:,column_name]).map(_compa0)
    plt.figure()
    plt.plot(temp[yname],label='median_5',linewidth=3,c='b')
    plt.scatter(x=temp.loc[temp['isdayu0'] > 1,yname].index,\
                y=temp.loc[temp['isdayu0'] > 1,yname],\
                c=["r"],marker='^',label="up",linewidths=10)
    plt.scatter(x=temp.loc[temp['isdayu0'] == 1,yname].index,\
                y=temp.loc[temp['isdayu0'] == 1,yname],\
                c=["k"],marker='o',label="equal",linewidths=10)
    plt.scatter(x=temp.loc[temp['isdayu0'] < 1,yname].index,\
                y=temp.loc[temp['isdayu0'] < 1,yname],\
                c=["g"],marker='v',label="down",linewidth=10)
    plt.title('%s - self.Tick_change'%(column_name))
    plt.xlabel("%s"%(column_name))
    plt.ylabel("self.Tick")
    plt.legend()
    plt.grid()
    
