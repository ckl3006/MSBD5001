# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 20:56:30 2018

@author: Leo
"""

import numpy as np
import pandas as pd
#from catboost import Pool, cv
from catboost import CatBoostRegressor
#from sklearn.model_selection import cross_validate
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
#from multiprocessing import Pool

if __name__ == '__main__':
    #read data from csv
    train_set = pd.read_csv("train.csv")
    test_set = pd.read_csv("test.csv")
    
    #for train data, separate the input (X) and output(Y) in 2 dataframes
    train_set_Y = train_set['time']
    
    #feature engineering
    #pick columns penalty, n_jobs, n_samples, n_features, n_samples, n_features, n_classes
    train_set_X = train_set.iloc[:, [1,6,7,8,9]]
    
    #estimate the iterations required to converge early stopping in "complexity of svm model'
    train_set_X['complexity_of_svm_model'] = train_set.iloc[:,7]*train_set.iloc[:,7]*train_set.iloc[:,8]/(np.log(train_set.iloc[:, 7])*175000)
    #pick coloumn of max iterations
    train_set_X['max_iter'] = train_set.iloc[:,4]
    #take the minimum value between "complexity of svm model' and max iteraton
    train_set_X['iterations'] = train_set_X[['complexity_of_svm_model','max_iter']].min(axis=1)
    
    #delete the two columns
    del train_set_X['complexity_of_svm_model']
    del train_set_X['max_iter']
    
    #replace -1 to 8 in n_jons (estimate the max thread is 8)
    train_set_X['n_jobs'].replace(-1, 8, inplace=True)
    
    train_set_X.info() #check purpose
    
    #check purpose for the train set dataframe
    train_set_X.to_csv("train_set_X.csv",encoding='utf-8')



    
    catgorical_features_indices = [0]

    
    #Train the model 5 times
    gbr = CatBoostRegressor(iterations = 3000, cat_features = catgorical_features_indices, depth=5, random_seed=1, loss_function="RMSE")
    gbr2 = CatBoostRegressor(iterations = 3000, cat_features = catgorical_features_indices, depth=5, random_seed=20, loss_function='RMSE')
    gbr3 = CatBoostRegressor(iterations = 3000, cat_features = catgorical_features_indices, depth=5, random_seed=300, loss_function='RMSE')
    gbr4 = CatBoostRegressor(iterations = 3000, cat_features = catgorical_features_indices, depth=5, random_seed=400, loss_function='RMSE')
    gbr5 = CatBoostRegressor(iterations = 3000, cat_features = catgorical_features_indices, depth=5, random_seed=500, loss_function='RMSE')
    
    #fit the 5 models
    predictor = gbr.fit(train_set_X, train_set_Y, cat_features= catgorical_features_indices,verbose=False)
    predictor2 = gbr2.fit(train_set_X, train_set_Y, cat_features= catgorical_features_indices,verbose=False)
    predictor3 = gbr3.fit(train_set_X, train_set_Y, cat_features= catgorical_features_indices,verbose=False)
    predictor4 = gbr4.fit(train_set_X, train_set_Y, cat_features= catgorical_features_indices,verbose=False)
    predictor5 = gbr5.fit(train_set_X, train_set_Y, cat_features= catgorical_features_indices,verbose=False)
    
    #feature engineering for test data (repeat similarly as the train data)
    X_test = test_set.iloc[:, [1,6,7,8,9]]
    
    X_test['complexity_of_svm_model'] = test_set.iloc[:,7]*test_set.iloc[:,7]*test_set.iloc[:,8]/(np.log(test_set.iloc[:, 7])*175000)
    X_test['max_iter'] = test_set.iloc[:,4]
    X_test['iterations'] = X_test[['complexity_of_svm_model','max_iter']].min(axis=1)
    
    del X_test['complexity_of_svm_model']
    del X_test['max_iter']
    
    # n_jobs replace -1 and 16 to 8 (estimate the max thread is 8)
    X_test['n_jobs'].replace(16, 8, inplace=True)
    X_test['n_jobs'].replace(-1, 8, inplace=True)

    #X_test.info() # - for testing purpose
   
    #predict the time(Y) for the testing data 5times
    Y_test = predictor.predict(X_test)
    Y_test_2 = predictor2.predict(X_test)
    Y_test_3 = predictor3.predict(X_test) 
    Y_test_4 = predictor4.predict(X_test)
    Y_test_5 = predictor5.predict(X_test)
    
        
    #Y_output = Y_test
    Y_output = (Y_test+Y_test_2+Y_test_3+Y_test_4+Y_test_5)/5
    #print(Y_output)
    
    #convert test output ndarray to pd series
    Y_output = pd.DataFrame(Y_output.reshape(-1, 1), columns = ["time"])
    
    
    #never use, backup purpose only - remove negative value in prediction to be at least 0.1
    Y_output.loc[Y_output['time']<0,'time'] = 0.1
    
    #output the result to the csv file
    Y_output.to_csv("Submission.csv",encoding='utf-8', index_label=["Id"])
    
    print("Finished!")
    
    