#test_xgboost1
#标准的库导入
import pandas as pd
import numpy as np
from datetime import datetime
import re
import matplotlib.pylab as plt
from math import sqrt
import os
from matplotlib.pyplot import rcParams
rcParams['figure.figsize']=15,6

# 预处理和划分数据
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 导入模型
import xgboost as xgb

#模型调参的工具
from sklearn.model_selection import cross_val_score,KFold
from sklearn.model_selection import GridSearchCV

#Error metrics
from sklearn.metrics import mean_squared_error, r2_score


#读取数据转换为dataframe格式
data_path = './data/data_104.dat'
data = np.memmap(data_path, dtype=np.float64)
data_len = int(data.shape[0] / 46)
data.resize([data_len,46])
df = pd.DataFrame(data)
X = df.iloc[:,2:-1]
y = df.iloc[:,-1]
print("data")
#划分测试集和训练集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# 拟合模型1
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
model.fit(X_train, y_train)

#测试1
pred_test_raw = model.predict(X_test)
MSE2 = mean_squared_error(pred_test_raw, y_test)
print("MSE2:", MSE2)
MSE=np.sum(np.power((y_test - pred_test_raw),2))/len(y_test)
R2=1-MSE/np.var(y_test)
print("MSE:",MSE)
print("R2:", R2)