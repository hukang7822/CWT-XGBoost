import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBRegressor
import joblib
import statsmodels.api as sm


#---------------------------------------------------------------------------------------------------------------------------------------- |
file_path_training1 = '' 
file_path_testining1 = ''  
file_out1 = ''  
trainModel_path1 = ''
#-----------------------------------------------------------------------------------------------------------------------------------------|


'''
----------------------------------------------------------------------------------------------------------------------------------------
1. Using XGBoost to predict target data                                                                                                 ;
----------------------------------------------------------------------------------------------------------------------------------------
'''

def XGBoost_train_cal(file_path_training, file_out, trainModel_path, test_size):
    f = open(file_out, 'w')

    data_training = pd.read_csv(file_path_training, sep=',', low_memory=False)  ###pandas读取数据，按照,间隔
    data_training.replace('NAN', np.nan, inplace=True)  ###将Igor的NAN值替换成pandas的NaN
    data_training.dropna(axis=0, how='any', inplace=True)
    num_column = data_training.shape[1]  ###数据列数

    column_names = data_training.columns.tolist()[1:-1]

    trainingData_X = pd.DataFrame(columns = range(num_column-2))
    trainingData_X = pd.DataFrame(trainingData_X, columns=column_names)

    testingData_X = pd.DataFrame(columns = range(num_column-2))
    testingData_X = pd.DataFrame(testingData_X, columns=column_names)

    #trainingData_Y = pd.Series().astype(float)
    #testingData_Y = pd.Series().astype(float)
    trainingData_Y = pd.DataFrame()
    testingData_Y = pd.DataFrame()

    m = 0

    for i in range(5):
        #random_number = random.randint(1, 100)
        #trainingData_X_temp, testingData_X_temp, trainingData_Y_temp, testingData_Y_temp = train_test_split(trainingData_X_original, trainingData_Y_original, test_size = test_size, random_state = random_number)

        num = int(data_training.shape[0] * test_size)

        n = num * (i + 1)
        testingData_temp = data_training.iloc[m:n, :].copy()
        trainingData_merge = data_training.merge(testingData_temp, how = 'left', indicator = True)
        trainingData_temp = trainingData_merge[trainingData_merge['_merge'] == 'left_only'].drop(columns = ['_merge'])


        trainingData_X_temp = trainingData_temp.iloc[:, 1:num_column - 1].copy()  
        trainingData_Y_temp = trainingData_temp.iloc[:, -1].copy()
        testingData_X_temp = testingData_temp.iloc[:, 1:num_column - 1].copy()  
        testingData_Y_temp = testingData_temp.iloc[:, -1].copy()
        DataTime = testingData_temp.iloc[:, 0].copy()

        trainingData_Y_temp = trainingData_Y_temp.to_frame()
        testingData_Y_temp = testingData_Y_temp.to_frame()

        trainingData_X = trainingData_X._append(trainingData_X_temp, ignore_index=True)
        testingData_X = testingData_X._append(testingData_X_temp, ignore_index=True)
        trainingData_Y = trainingData_Y._append(trainingData_Y_temp, ignore_index=True)
        testingData_Y = testingData_Y._append(testingData_Y_temp, ignore_index=True)

        m = n + 1

    model = XGBRegressor(learning_rate= 0.01,      
                         n_estimators=2508,
                         max_depth=9,
                         min_child_weight=1,
                         gamma=0.1,
                         colsample_bytree=0.6,
                         subsample=0.6,
                         reg_alpha=400,
                         reg_lambda= 1,
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

    res = model.predict(testingData_X.astype(float))

    Y = res
    X = pd.Series(testingData_Y[data_training.columns.tolist()[-1]].values).astype(float)
    Y = pd.Series(Y).astype(float)
    #X = pd.Series(X).astype(float)
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
    data1_res = testingData_Y[data_training.columns.tolist()[-1]].values.tolist()
    #DataTime_res = list(DataTime)


    for i in range(len(data1_res)):
        m = float(data1_res[i])
        data_res.append(m)
    for i in range(len(data_res)):
        #f.write('%10s,%10s,%10s\n' % (DataTime_res[i], data_res[i], pre_res[i]))
        f.write('%10s,%10s\n' % (data_res[i], pre_res[i]))
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
    XGBoost_training()
