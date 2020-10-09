# ts_predict

## 1 Neural Network

### 1.1 DeepAR

### 1.2 LSTM

运行
```
python main.py \
    -data_path=/data00/yujunshuai/code/ts_predict/data/data_289.dat \
    -lr=0.001 \
    -epochs=100 \
    -batch_size=512 \
    -output_size=1 \
    -save_dir=experients/lstm-v1 \
    -model_name=lstm-v1 \
    -device=7 \
    -log_interval=25
```

|  参数   | 训练集指标  | 测试集指标  |
|  ----  | ----  | --- |
| window_size = 100 | ** | ** | 


## 2 Machine Learning


|  参数   | 训练集指标  | 测试集指标  |
|  ----  | ----  | --- |
| xgboost | 0.09158208542535884 | -0.02345814391564982 | 
| LinearRegression | 0.007551351390086336 | 0.0041194783593629936 | 
| SVR | -5.967226016540408 | -5.093447881856339 | 
| RandomForest | 0.9246995206961224 | -0.09154779055053286 |
| Adaboost | -0.1268407033126091 | -0.14848386651548573 |
| GradientBoosting | 0.02048480080542059 | 0.00010562969933525235 |
| ExtraTree | 0.9999977427047394 | -1.53035696525583 |
| 

****
