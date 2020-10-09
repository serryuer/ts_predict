import os, sys, json
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import numpy
from sklearn.tree import DecisionTreeRegressor

def get_score(path):
    data = np.memmap(path, dtype='float64', mode='r')
    data = np.array(data.reshape((int(len(data)/46), 46)))
    bad_row = []
    for i in range(data.shape[0]):
        if data[i, 0]==0:
            bad_row.append(i)
    data = np.delete(data, bad_row, axis=0)
    X = data[:, 2:-1]
    Y = data[:, -1].reshape(-1, 1)
    split_at = int(X.shape[0]*0.8)
    X_tr = X[:split_at,:]
    X_te = X[split_at:,:]
    Y_tr = Y[:split_at,:]
    Y_te = Y[split_at:,:]
    model = DecisionTreeRegressor(max_depth=2)
    model.fit(X_tr, Y_tr)
    
    plt.scatter([i for i in range(X_tr.shape[0])], Y_tr, c='lightblue')
    plt.plot([i for i in range(X_tr.shape[0])], model.predict(X_tr), color='red', linewidth=2)  
    plt.tight_layout()
    plt.savefig('/data00/yujunshuai/code/ts_predict/ML/figures/gbdt.png', dpi=300)
    plt.show()
    
    return model.score(X_tr, Y_tr), model.score(X_te, Y_te)

def test():
    result = pd.DataFrame(columns=['train', 'test'])
    for i in range(1, 301):
        try:
            data_str = 'data_' + str(i).zfill(3) + '.dat'
            path = '../data/' + data_str
            score_tr, score_te = get_score(path)
            result.loc[i, 'train'] = score_tr
            result.loc[i, 'test'] = score_te
        except:
            pass
    result.to_csv('linear.csv')
    print(result.astype(float).describe())

if __name__ == '__main__':
    # test()
    sys.argv.append('/data00/yujunshuai/code/ts_predict/data/data_289.dat')
    print(get_score(sys.argv[1]))