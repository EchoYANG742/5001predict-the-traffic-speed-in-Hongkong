#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: YangYi
# studentid:20710742

import pandas as pd
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
import numpy as np

df =pd.read_csv("/content/train.csv",parse_dates=['date']) #将csv文件中的date创建为dataframe文件格式
dftest=pd.read_csv("/content/test.csv",parse_dates=['date'])

def split_date(df,column):
  df[column+'_year'] = df[column].apply(lambda x: x.year)
  df[column+'_month'] = df[column].apply(lambda x: x.month)
  df[column+'_day'] = df[column].apply(lambda x: x.day)
  df[column+'_hour'] = df[column].apply(lambda x: x.hour)

split_date(df,'date')
split_date(dftest,'date')
df['dayofweek']=df['date'].dt.dayofweek  #增加星期的判断
dftest['dayofweek']=dftest['date'].dt.dayofweek

#print(df.head())
#print(dftest.head())

def trainandTest(X_train, y_train, X_test):
    # XGBoost训练过程 适当增加max_depth和n_estimators可提高精确度
    model = xgb.XGBRegressor(max_depth=10, learning_rate=0.05, n_estimators=500, silent=False, objective='reg:gamma')
    model.fit(X_train, y_train)

    # 对测试集进行预测
    ans = model.predict(X_test)

    ans_len = len(ans)
    id_list = np.arange(0, 5000)
    data_arr = []
    for row in range(0, ans_len):
        data_arr.append([int(id_list[row]), ans[row]])
    np_data = np.array(data_arr)   #np_data类型是numpy.ndarray
    
    # 写入文件
    pd_data = pd.DataFrame(np_data,columns=['id', 'speed'])
    pd_data['id'] = pd_data['id'].astype(np.int32)
    pd_data['speed'] = pd_data['speed'].astype(np.float64)
    print(pd_data)
    pd_data.to_csv('submityy.csv', index=None)


if __name__ == '__main__':
    X_train = df.iloc[:, 3:].values
    y_train= df.iloc[:, 2].values
    X_test = dftest.iloc[:, 2:].values
    
    trainandTest(X_train, y_train, X_test)