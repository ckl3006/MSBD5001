# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 20:22:30 2018

@author: Leo
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

if __name__ == '__main__':
    #read data from csv
    train_set = pd.read_csv("train.csv")
    test_set = pd.read_csv("test.csv")
    
    #for train data, separate the input (X) and output(Y) in 2 dataframes
    Y_train = train_set['time']
    original_X_train = train_set.iloc[:, 0:13]
    
    # Apply one hot encoding to train data(X)
    X_train = pd.get_dummies(original_X_train) 
    
    #print(X_train.shape) #check purpose - it should contain 16 columns
    
    # X_train.info() #check purpose
    
    #train the gradient boosting regressor  
    gbr = GradientBoostingRegressor(n_estimators=40)
    predictor = gbr.fit(X_train, Y_train)
    print ("R-squared for Train: %.4f" %gbr.score(X_train, Y_train))
    
    #predict Y
    extracted_test_data_input = test_set.iloc[:, 1:]
    # extracted_test_label = test_set.iloc[:, 0] - for testing purpose

    # Apply one hot encoding to train data(X)
    X_test = pd.get_dummies(extracted_test_data_input)
    
    #print("Testing on the dimensions of test data") - for testing purpose
    #print(X_test) - for testing purpose
    #X_test.info() - for testing purpose
    
    #predict the time(Y) for the testing data
    Y_test = predictor.predict(X_test)
    
    #convert test output ndarray to pd series
    Y_test = pd.DataFrame(Y_test.reshape(-1, 1), columns = ["time"])
    
    #print(type(Y_test)) - for testing purpose
    #print(Y_test.shape) - for testing purpose
    #print(Y_test) - for testing purpose
        
    #output the result to the csv file
    output = Y_test
    output.to_csv("Submission.csv",encoding='utf-8', index_label=["Id"])
    
    print("Finished!")
    
    