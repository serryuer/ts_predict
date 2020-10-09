import os, sys, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

model_maps = {}

grid_search = False

###########具体方法选择##########
# ####3.1决策树回归####
# from sklearn import tree
# cv_params = {'max_depth': np.linspace(1, 10, 10, dtype=int)}
# model_maps['decision_tree'] = tree.DecisionTreeRegressor(max_depth=2) if not grid_search else GridSearchCV(tree.DecisionTreeRegressor(max_depth=6), cv_params, verbose=2, refit=True, cv=5, n_jobs=10)
# ####3.2线性回归####
# from sklearn import linear_model
# model_maps['linear'] = linear_model.LinearRegression()
# ####3.3SVM回归####
# from sklearn import svm
# model_maps['svr'] = svm.SVR()
# ####3.4KNN回归####
# from sklearn import neighbors
# model_maps['knn'] = neighbors.KNeighborsRegressor()
####3.5随机森林回归####
from sklearn import ensemble
model_maps['random_forest'] = ensemble.RandomForestRegressor(n_estimators=10)#这里使用20个决策树
####3.6Adaboost回归####
from sklearn import ensemble
model_maps['adaboost'] = ensemble.AdaBoostRegressor(n_estimators=50)#这里使用50个决策树
####3.7GBRT回归####
from sklearn import ensemble
model_maps['gradient_boosting'] = ensemble.GradientBoostingRegressor(n_estimators=100)#这里使用100个决策树
####3.8Bagging回归####
from sklearn.ensemble import BaggingRegressor
model_maps['bagging'] = BaggingRegressor()
####3.9ExtraTree极端随机树回归####
from sklearn.tree import ExtraTreeRegressor
model_maps['extra_tree'] = ExtraTreeRegressor()

def get_score(data_path, model_name):
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
    X_tr = X[:split_at,:]
    X_te = X[split_at:,:]
    Y_tr = Y[:split_at,:]
    Y_te = Y[split_at:,:]
    model = model_maps[model_name]
    model.fit(X_tr, Y_tr)
    
    if grid_search:
        print("参数的最佳取值：:", model.best_params_)
        print("最佳模型得分:", model.best_score_)
    
    plt.scatter([i for i in range(X_tr.shape[0])], Y_tr, c='lightblue')
    plt.plot([i for i in range(X_tr.shape[0])], model.predict(X_tr), color='red', linewidth=2)  
    plt.tight_layout()
    plt.savefig(f'/data00/yujunshuai/code/ts_predict/ML/figures/{model_name}.png', dpi=300)
    plt.show()
    
    return model.score(X_tr, Y_tr), model.score(X_te, Y_te)

if __name__ == '__main__':
    sys.argv.append('/data00/yujunshuai/code/ts_predict/data/data_289.dat')
    sys.argv.append('decision_tree')
    
    for model_name in model_maps.keys():
        print(model_name, end='\t')
        print(get_score(data_path=sys.argv[1], model_name=model_name))