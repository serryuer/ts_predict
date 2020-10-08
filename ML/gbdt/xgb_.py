import os, sys, json
import pandas as pd
import numpy as np
import re
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import cross_val_score,KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score


cv_params = {
    # 'n_estimators': np.linspace(10, 200, 10, dtype=int),
    'max_depth': np.linspace(1, 10, 10, dtype=int),
    # 'min_child_weight': np.linspace(1, 10, 10, dtype=int),
    # 'gamma': np.linspace(0, 1, 10),
    # 'gamma': np.linspace(0, 0.1, 11),
    # 'subsample': np.linspace(0, 1, 11),
    # 'subsample': np.linspace(0.9, 1, 11),
    # 'colsample_bytree': np.linspace(0, 1, 11)[1:],
    # 'reg_lambda': np.linspace(0, 100, 11),
    # 'reg_lambda': np.linspace(40, 60, 11),
    # 'reg_alpha': np.linspace(0, 10, 11),
    # 'reg_alpha': np.linspace(0, 1, 11),
    # 'eta': np.logspace(-2, 0, 10),
    }

other_params = {
        'n_estimators': 31, 
        'eta': 0.3, 
        # 'gamma': 0.11111, 
        # 'max_depth': 3, 
        # 'min_child_weight': 1,
        # 'colsample_bytree': 0.1, 
        # 'colsample_bylevel': 1, 
        # 'subsample': 0.9, 
        # 'reg_lambda': 70, 
        # 'reg_alpha': 0,
        # 'seed': 33
    }


def train(data_path):
    data = np.memmap(data_path, dtype='float64', mode='r')
    data = np.array(data.reshape((int(len(data)/46), 46)))
    bad_row = []
    for i in range(data.shape[0]):
        if data[i, 0]==0:
            bad_row.append(i)
    data = np.delete(data, bad_row, axis=0)
    X = data[:, 2:-1]
    Y = data[:, -1].reshape(-1, 1)
    split_at = int(X.shape[0]*0.8)
    X_train = X[:split_at,:]
    X_test = X[split_at:,:]
    Y_train = Y[:split_at,:]
    Y_test = Y[split_at:,:]
    model = xgb.XGBRegressor(**other_params)
    gs = GridSearchCV(model, cv_params, verbose=2, refit=True, cv=5, n_jobs=60)
    gs.fit(X_train, Y_train)  # X为训练数据的特征值，y为训练数据的label
    # 性能测评
    print("参数的最佳取值：:", gs.best_params_)
    print("最佳模型得分:", gs.best_score_)
    print(model.get_params())
    model.fit(X_train, Y_train)
    
    return r2_score(Y_train, model.predict(X_train)), r2_score(Y_test, model.predict(X_test))

if __name__ == '__main__':
    sys.argv.append('/data00/yujunshuai/code/ts_predict/data/data_289.dat')
    print(train(sys.argv[1]))
    

