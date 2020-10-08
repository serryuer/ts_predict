# ts_predict

## 1 Neural Network

## 2 Machine Learning

### 2.1 gbdt

#### 2.1.1 xgboost

|  参数   | 训练集指标  | 测试集指标  |
|  ----  | ----  | --- |
| objective='reg:squarederror', n_estimators=100 | 0.09158208542535884 | -0.02345814391564982 | 

#### 2.1.2 lightgbm

#### 2.1.3 catboost

#### 2.1.4 DecisionTreeRegression

|  参数   | 训练集指标  | 测试集指标  |
|  ----  | ----  | --- |
| DecisionTreeRegressor(max_depth=3)  | 0.0034978530750585657 | 0.0021135110353691644 | 


## 2.2 LR

|  参数   | 训练集指标  | 测试集指标  |
|  ----  | ----  | --- |
| / | 0.007551351390086336 | 0.0041194783593629936 | 

## 2.3 SVM

|  参数   | 训练集指标  | 测试集指标  |
|  ----  | ----  | --- |
| LinearSVR(epsilon=1.5) | -0.00022916164456820987 | -0.00011044924201764061 | 
| SVR(kernel="poly", degree=2, C=100, epsilon=0.1) | -5.967226016540408 | -5.093447881856339 | 

## 2.4 
****
