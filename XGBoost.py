'''
python code to combined multiply files
From: Kang Hu, Nanjing University of Information Science and Technology, Hong Liao group
Author: Kang Hu, NUIST
Written: 2024-04-11
Version 1.0
email: 200060@nuist.edu.cn
'''

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBRegressor
import joblib
import statsmodels.api as sm


#---------------------------------------------------------------------------------------------------------------------------------------- |
file_path_training1 = '/Volumes/HK/Team_Group/Sihan/output70percent.csv' 
file_path_testining1 = '/Volumes/HK/Team_Group/Sihan/output30percent.csv'  
file_out1 = '/Volumes/HK/Team_Group/Sihan/XGBoost-total-PM1.csv'  
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

    model = XGBRegressor(learning_rate= 0.1,     
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
    data_training = pd.read_csv(file_path_training, sep=',', low_memory=False)  
    data_training.replace('NAN', np.nan, inplace=True)  
    data_training.dropna(axis=0, how='any', inplace=True)
    num_column = data_training.shape[1]  
    trainingData_X = data_training.iloc[:, 1:num_column - 1]  
    trainingData_Y = data_training.iloc[:, -1]

    data_testing = pd.read_csv(file_path_testining, sep=',', low_memory=False)  
    data_testing.replace('NAN', np.nan, inplace=True)  
    data_testing.dropna(axis=0, how='any', inplace=True)
    testData_X = data_testing.iloc[:, 1:num_column - 1]  
    testData_Y = data_testing.iloc[:, -1]
    DataTime = data_testing.iloc[:, 0]

    return DataTime,trainingData_X, trainingData_Y, testData_X, testData_Y


######################################
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



if __name__ == '__main__':
    XGBoost_predicting()
