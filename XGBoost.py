'''
python code to combined multiply files
From: Kang Hu, Nanjing University of Information Science and Technology, Hong Liao group
Author: Kang Hu, NUIST
Written: 2024-04-11
Version 1.0
email: 200060@nuist.edu.cn
First column is date; Second column is original data; Last column is target data
'''

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBRegressor
import joblib
import statsmodels.api as sm

#全局只需要在这里输入文件路径
#---------------------------------------------------------------------------------------------------------------------------------------- |
file_path_training1 = '/Volumes/HK/Team_Group/Sihan/output70percent.csv'  #######这里放置训练文件路径  ｜
file_path_testining1 = '/Volumes/HK/Team_Group/Sihan/output30percent.csv'  #######这里放置测试文件路径                        ｜
file_out1 = '/Volumes/HK/Team_Group/Sihan/XGBoost-total-PM1.csv'  #######这里放置输出文件路径                  ｜
trainModel_path1 = '/Volumes/HK/Team_Group/Sihan/output-Residual.m'
#-----------------------------------------------------------------------------------------------------------------------------------------|


'''
----------------------------------------------------------------------------------------------------------------------------------------
1. Using XGBoost to predict target data                                                                                                 ;
----------------------------------------------------------------------------------------------------------------------------------------
'''

def XGBoost_train_cal(file_path_training, file_path_testining,file_out, trainModel_path):
    f = open(file_out, 'w')

    DataTime, trainingData_X, trainingData_Y, testData_X, testData_Y = Data_read(file_path_training, file_path_testining)

    model = XGBRegressor(learning_rate= 0.1,      ###学习率
                         n_estimators=1000,
                         max_depth=11,
                         min_child_weight=3,
                         gamma=0.1,
                         colsample_bytree=0.8,
                         subsample=0.6,
                         reg_alpha=0.085,
                         reg_lambda=50,
                         nthread=4,
                         scale_pos_weight=1,
                         objective = 'reg:squarederror',
                         eval_metric=['rmse'],
                         seed=27,
                         booster='gbtree',
    )


    model.fit(trainingData_X.astype(float),trainingData_Y.astype(float))

    joblib.dump(model, trainModel_path)

    FI = pd.Series(model.get_booster().get_fscore())

    fet_imp = "RandomForest Feature importance are (%s)" % (str(FI))

    f.write(fet_imp)
    f.write('\n')

    res = model.predict(testData_X.astype(float))

    Y = res
    X = testData_Y.astype(float)
    Y = pd.Series(Y).astype(float)
    X = pd.Series(X).astype(float)
    Y = Y.reset_index(drop=True)
    Y = pd.concat([Y], axis=1)
    X = X.reset_index(drop=True)
    X = pd.concat([X], axis=1)
    sqrd_xy = pd.concat([X, Y], axis=1)
    sqrd_xy.dropna(axis=0, how='any', inplace=True)
    modell = sm.OLS(sqrd_xy.iloc[:, 0], sm.add_constant(sqrd_xy.iloc[:, 1]), hasconst=True).fit()
    R2 = modell.rsquared
    print(R2)

    data_res = []  # Transfer list data to eval
    pre_res = list(res)
    data1_res = list(trainingData_Y)
    DataTime_res = list(DataTime)


    for i in range(len(data1_res)):
        m = float(data1_res[i])
        data_res.append(m)
    for i in range(len(data_res)):
        f.write('%10s,%10s,%10s\n' % (DataTime_res[i], data_res[i], pre_res[i]))
    f.close()

def XGBoost_predict_cal(file_path_training, file_path_testining,file_out, trainModel_path):
    f = open(file_out, 'w')

    DataTime, trainingData_X, trainingData_Y, testData_X, testData_Y = Data_read(file_path_training, file_path_testining)

    model = joblib.load(trainModel_path)

    res = model.predict(testData_X.astype(float))

    data_res = []  # Transfer list data to eval
    pre_res = list(res)
    data1_res = list(testData_Y)
    DataTime_res = list(DataTime)

    for i in range(len(data1_res)):
        m = float(data1_res[i])
        data_res.append(m)
    for i in range(len(data_res)):
        f.write('%10s,%10s,%10s\n' % (DataTime_res[i], data_res[i], pre_res[i]))
    f.close()


def Data_read(file_path_training,file_path_testining):
    data_training = pd.read_csv(file_path_training, sep=',', low_memory=False)  ###pandas读取数据，按照,间隔
    data_training.replace('NAN', np.nan, inplace=True)  ###将Igor的NAN值替换成pandas的NaN
    data_training.dropna(axis=0, how='any', inplace=True)
    num_column = data_training.shape[1]  ###数据列数
    trainingData_X = data_training.iloc[:, 1:num_column - 1]  ###提取测试集（直到最后一列的前一列）数据70%
    trainingData_Y = data_training.iloc[:, -1]

    data_testing = pd.read_csv(file_path_testining, sep=',', low_memory=False)  ###pandas读取数据，按照,间隔
    data_testing.replace('NAN', np.nan, inplace=True)  ###将Igor的NAN值替换成pandas的NaN
    data_testing.dropna(axis=0, how='any', inplace=True)
    testData_X = data_testing.iloc[:, 1:num_column - 1]  ###提取测试集（直到最后一列的前一列）数据70%
    testData_Y = data_testing.iloc[:, -1]
    DataTime = data_testing.iloc[:, 0]

    return DataTime,trainingData_X, trainingData_Y, testData_X, testData_Y


######################################只要调用这个程序，可实现预测
def XGBoost_training():
    file_path_testining = file_path_testining1
    file_out = file_out1
    trainModel_path = trainModel_path1
    XGBoost_train_cal(file_path_testining, file_path_testining, file_out, trainModel_path)

def XGBoost_predicting():
    file_path_training = file_path_training1
    file_path_testining = file_path_testining1
    file_out = file_out1
    trainModel_path = trainModel_path1
    XGBoost_predict_cal(file_path_training, file_path_testining, file_out, trainModel_path)
'''
----------------------------------------------------------------------------------------------------------------------------------------
2. Adjusting the input Parameters                                                                                                       ;
----------------------------------------------------------------------------------------------------------------------------------------
'''
###下面这个不要动，也不要管
#---------------------------------------------------------------------------------------------------------------------------------------- |
def modelfit(model, XGB_trainingData,cols, target):                                                                                      #|
    xgb_param = model.get_xgb_params()                                                                                                   #|
    xgb_train = xgb.DMatrix(XGB_trainingData[cols].values, XGB_trainingData[target].values)                                              #|
    curesult = xgb.cv(xgb_param,xgb_train, num_boost_round= model.get_params()['n_estimators'], nfold = 10,                              #|
                      early_stopping_rounds= 20)                                                                                         #|
    model.fit(XGB_trainingData[cols].astype(float),XGB_trainingData[target].astype(float))                                                                           #|
    print (curesult.shape[0])                                                                                                            #|
                                                                                                                                         #|
                                                                                                                                         #|
def Adjusting_path():                                                                                                                    #|
    file_path_training = file_path_training1                                                                                             #|
    data_training = pd.read_csv(file_path_training, sep=',', low_memory=False)                                                                             #|
    data_training.replace('NAN', np.nan, inplace=True)                                                                                   #|
    data_training.dropna(axis=0, how='any', inplace=True)                                                                                #|
    testTemp = data_training.iloc[:, 1:].copy()                                                                                          #|
    target = [x for x in data_training.columns][-1]                                                                                      #|
    cols = [x for x in testTemp.columns if x not in target]                                                                              #|
    return testTemp,cols,target                                                                                                          #|
#---------------------------------------------------------------------------------------------------------------------------------------- |


def n_estimators_adjust():
    XGB_trainingData,cols,target = Adjusting_path()
    xgb_num_round = XGBRegressor(learning_rate = 0.1,
                                 n_estimators = 50000,
                                 max_depth = 5,
                                 min_child_weight = 1,
                                 gamma = 0,
                                 subsample = 0.8,
                                 colsample_bytree = 0.8,
                                 nthread = 4,
                                 scale_pos_weight = 1,
                                 eval_metric= ['rmse'],
                                 seed = 27,
                                 booster='gbtree',
                                 )
    modelfit(xgb_num_round,XGB_trainingData,cols,target)


def max_depth_min_child_weight_adjust():
    XGB_trainingData, cols, target = Adjusting_path()

    param_grid = {'max_depth': range(5,15,2),
                  'min_child_weight': range(1,10,2)}

    model = XGBRegressor(learning_rate = 0.1,
                         n_estimators = 13421,
                         gamma = 0,
                         subsample = 0.8,
                         colsample_bytree = 0.8,
                         nthread=4,
                         scale_pos_weight=1,
                         eval_metric=['rmse'],
                         seed=27,
                         booster='gbtree',
                         )
    gsearch1 = GridSearchCV(estimator=model, param_grid= param_grid, n_jobs= -1, cv= 5)

    gsearch1.fit(XGB_trainingData[cols].astype(float),XGB_trainingData[target].astype(float))

    print ('最佳参数是:', gsearch1.best_params_)
    print ('得分：', gsearch1.best_score_)

def gamma_adjust():
    XGB_trainingData, cols, target = Adjusting_path()

    param_grid = {'gamma': [i/10.0 for i in range(0,5)]}

    model = XGBRegressor(learning_rate = 0.1,
                         n_estimators = 13421,
                         max_depth=11,
                         min_child_weight=3,
                         subsample = 0.8,
                         colsample_bytree = 0.8,
                         nthread=4,
                         scale_pos_weight=1,
                         eval_metric=['rmse'],
                         seed=27,
                         booster = 'gbtree',
                         )
    gsearch1 = GridSearchCV(estimator=model, param_grid= param_grid, n_jobs= -1, cv= 5)

    gsearch1.fit(XGB_trainingData[cols].astype(float),XGB_trainingData[target].astype(float))

    print ('最佳参数是:', gsearch1.best_params_)
    print ('得分：', gsearch1.best_score_)

def subsample_colsample_bytree_adjust():
    XGB_trainingData, cols, target = Adjusting_path()

    param_grid = {'subsample': [i/10.0 for i in range(6,10)],
                  'colsample_bytree': [i/10.0 for i in range(6,10)]}

    model = XGBRegressor(learning_rate = 0.1,
                         n_estimators = 13421,
                         max_depth=11,
                         min_child_weight=3,
                         gamma=0.1,
                         nthread=4,
                         scale_pos_weight=1,
                         eval_metric=['rmse'],
                         seed=27,
                         booster='gbtree',
                         )
    gsearch1 = GridSearchCV(estimator=model, param_grid= param_grid, n_jobs= -1, cv= 5)

    gsearch1.fit(XGB_trainingData[cols].astype(float),XGB_trainingData[target].astype(float))

    print ('最佳参数是:', gsearch1.best_params_)
    print ('得分：', gsearch1.best_score_)

def reg_alpha_adjust():
    XGB_trainingData, cols, target = Adjusting_path()

    param_grid = {'reg_alpha': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000],}
    param_grid = {'reg_alpha': [0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 1.0, 1.005, 1.01, 1.015, 1.02, 1.025, 1.03], }

    model = XGBRegressor(learning_rate = 0.1,
                         n_estimators = 13421,
                         max_depth=11,
                         min_child_weight=3,
                         gamma=0.1,
                         colsample_bytree=0.8,
                         subsample=0.6,
                         nthread=4,
                         scale_pos_weight=1,
                         eval_metric=['rmse'],
                         seed=27,
                         booster='gbtree',
                         )
    gsearch1 = GridSearchCV(estimator=model, param_grid= param_grid, n_jobs= -1, cv= 5)

    gsearch1.fit(XGB_trainingData[cols].astype(float),XGB_trainingData[target].astype(float))

    print ('最佳参数是:', gsearch1.best_params_)
    print ('得分：', gsearch1.best_score_)

def reg_lambda_adjust():
    XGB_trainingData, cols, target = Adjusting_path()

    #param_grid = {'reg_lambda': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 1500, 2000, 2500],}
    param_grid = {'reg_lambda': [44, 44.5, 45, 45.5, 46, 46.5, 47, 47.5, 48, 48.5, 49, 49.5, 50, 50.5, 51, 51.5, 52, 52.5, 53, 53.5, 54, 54.5, 55, 55.5], }

    model = XGBRegressor(learning_rate = 0.1,
                         n_estimators=13421,
                         max_depth=11,
                         min_child_weight=3,
                         gamma=0.1,
                         colsample_bytree=0.8,
                         subsample=0.6,
                         reg_alpha = 0.085,
                         nthread=4,
                         scale_pos_weight=1,
                         eval_metric=['rmse'],
                         seed=27,
                         booster='gbtree',
                         )
    gsearch1 = GridSearchCV(estimator=model, param_grid= param_grid, n_jobs= -1, cv= 5)

    gsearch1.fit(XGB_trainingData[cols].astype(float),XGB_trainingData[target].astype(float))

    print ('最佳参数是:', gsearch1.best_params_)
    print ('得分：', gsearch1.best_score_)


##########降低学习率，learning_rate = 0.01



if __name__ == '__main__':
    XGBoost_predicting()