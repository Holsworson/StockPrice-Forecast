# -*- coding: utf-8 -*-

import pandas as pd

def SumV(data, begin_number=1, end_number=4):
    '''
    计算买卖盘口的Volume  和  以及差值
    end_number 取 2、3
    '''
    end_number = min(end_number,10)
    
    AskColumnName = "AskSumV" + str(begin_number) + str(end_number)
    BidColumnName = "BidSumV" + str(begin_number) + str(end_number)
    deltaColumnName = "deltaSumV" + str(begin_number) + str(end_number)
    
    data.loc[:,AskColumnName] = pd.Series([0.0] * len(data))
    data.loc[:,BidColumnName] = pd.Series([0.0] * len(data))
    for i in range(begin_number,end_number):
        data.loc[:,AskColumnName] += data['AskV' + str(i)]
        data.loc[:,BidColumnName] += data['BidV' + str(i)]   
        
    data[deltaColumnName] = data[AskColumnName] - data[BidColumnName]

    
    
    
    

    
    
