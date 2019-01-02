# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 12:39:18 2018

@author: vwzheng
"""

import os
import pandas as pd
import numpy as np
from math import *

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 
              'sqft_living15':float, 'grade':int, 'yr_renovated':int, 
              'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 
              'sqft_lot15':float, 'sqft_living':float, 'floors':float, 
              'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 
              'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

filePath = 'D:/Downloads/vivienne/ML/Regression_UW/kc_house_data.csv'
sales = pd.read_csv(filePath, dtype=dtype_dict)

#Take a data set, a list of features to be used as inputs, 
#and a name of the output
def get_data(data, features, output):
    #add a constant column to a data_Frame if it doesn't have
    data['constant'] = 1 
    #prepend variable 'constant' to the features list
    if 'constant' not in features:
        features.insert(0, 'constant')
    #select the columns of data_Frame given by the ‘features’ list 
    feature_matrix = data[features]
    #assign the column of data associated with the target to the ‘output’
    output_array = data[output]
    return (feature_matrix, output_array)

#Predicted output is dot product between features matrix and weights     
def predict_outcome(feature_matrix, weights):
    predictions = np.dot(feature_matrix, weights) 
    return predictions

#Normalize columns of a given feature matrix
def normalize_features(feature_matrix):
    #Compute 2-norms of columns
    norms = np.linalg.norm(feature_matrix, axis=0) 
    #Normalize columns and perform element-wise division
    normalized_features = feature_matrix/norms
    return (normalized_features, norms)

'''
Coordinate Descent
Lasso Cost Function =
    SUM[ (prediction - output)^2 ] + lambda*( |w[1]| + ... + |w[k]|).
argmin_{w[i]} [SUM[(prediction - output)^2] + lambda*(|w[1]| + ... + |w[k]|)]
       ┌ (ro[i] + lambda/2)    if ro[i] < -lambda/2
w[i] = ├ 0                     if -lambda/2 <= ro[i] <= lambda/2
       └ (ro[i] - lambda/2)    if ro[i] > lambda/2    
ro[i] = SUM[ [feature_i]*(output - prediction + w[i]*[feature_i]) ].
w[0] = ro[i]
Effect of CD
ro[i] = SUM[ [feature_i]*(output - prediction + w[i]*[feature_i]) ]       
 '''

#Single Coordinate Descent Step
def lasso_coordinate_descent_step(i, feature_matrix, output, weights, 
                                  l1_penalty):
    #compute prediction
    prediction = predict_outcome(feature_matrix, weights)
    #compute ro[i]=SUM[[feature_i]*(output-prediction+weight[i]*[feature_i])]
    ro_i = (feature_matrix.iloc[:,i]*(output-prediction+
                                      weights[i]*feature_matrix.iloc[:,i])
            ).sum()
    
    if i == 0: # intercept -- do not regularize
        new_weight_i = ro_i
    elif ro_i < -l1_penalty/2.:
        new_weight_i = ro_i + l1_penalty/2
    elif ro_i > l1_penalty/2.:
        new_weight_i = ro_i - l1_penalty/2
    else:
        new_weight_i = 0. 
    return new_weight_i

# should print 0.425558846691
print(lasso_coordinate_descent_step(1, pd.DataFrame(
      [[3./sqrt(13), 1./sqrt(10)], 
       [2./sqrt(13), 3./sqrt(10)]]), np.array([1., 1.]), 
      np.array([1., 4.]), 0.1))      
    
#Cyclical Coordinate Descent    
def lasso_cyclical_coordinate_descent(feature_matrix, output, 
                                      initial_weights, l1_penalty, 
                                      tolerance):
    d = feature_matrix.shape[1] #number of columns of features
    weights = np.array(initial_weights)
    changes = np.array(initial_weights)*0.0
    converged = False
    
    #For each iteration:
    while not converged:
        for i in range(d):
            #As you loop over features in order & perform coordinate descent, 
            #measure how much each coordinate changes.
            weight = lasso_coordinate_descent_step(i, feature_matrix, output, 
                                                   weights, l1_penalty)
            changes[i] = np.abs(weights[i] - weight)
            weights[i] = weight
            
        #After the loop, if the maximum change across all coordinates falls 
        #below the tolerance, stop. Otherwise, go back to the previous step.
        change_max = max(changes)
        if change_max < tolerance:
            converged = True
            
    return weights


#Learn the weights on the sales dataset
#Extract the feature matrix and the output array from the house dataframe
#Then normalize the feature matrix    
(simple_features, simple_output) = get_data(sales, ['sqft_living', 
                                                    'bedrooms'], 'price')  
norm_simple_features, simple_norms = normalize_features(simple_features)

#Effect of L1 penalty 
 '''
Recall that, whenever ro[i] falls between -l1_penalty/2 and l1_penalty/2, 
the corresponding weight w[i] is sent to zero. 
Now suppose we were to take one step of coordinate descent on either 
feature 1 or feature 2. 
What range of values of l1_penalty would not set w[1] zero, 
but would set w[2] to zero, if we were to take a step in that coordinate?
-lambda/2 <= ro[2] <= lambda/2 and ro[1] < -lambda/2 or ro[1] > lambda/2
Therefore, lambda >= 2*ro[2] and lambda < 2*ro[1] 
What range of values of l1_penalty would set both w[1] and w[2] to zero, 
if we were to take a step in that coordinate? Therefore, lambda >= 2*ro[1]
So we can say that ro[i] quantifies the significance of the i-th feature: 
the larger ro[i] is, the more likely for the i-th feature to be retained.
'''
weights = [1.,4.,1.]
ros = np.zeros(len(weights))
for i in range(len(weights)):
    ros[i] =(norm_simple_features.iloc[:,i]*(simple_output - predict_outcome(
                                             norm_simple_features, weights) +  
                                             weights[i]*
                                             norm_simple_features.iloc[:,i])
              ).sum()
print(ros, ros*2)
#[ 79400300.0145229   87939470.82325175  80966698.66623947] 
#[  1.58800600e+08   1.75878942e+08   1.61933397e+08]
#1.62e8<lambda<1.76e8 for w[2]=0
#lambda>1.76e8 for w[1]=0 and w[2]=0


initial_weights = [0., 0., 0.]
l1_penalty = 1e7
tolerance = 1.0
simple_weights = lasso_cyclical_coordinate_descent(norm_simple_features, 
                                                   simple_output, 
                                                   initial_weights, 
                                                   l1_penalty, tolerance)
RSS_simple = ((predict_outcome(norm_simple_features, simple_weights)-
               simple_output)**2).sum()

print(simple_features.columns, simple_weights) 
#bedrooms weights 0 at convergence

trainPath = 'D:/Downloads/vivienne/ML/Regression_UW/kc_house_train_data.csv'
testPath = 'D:/Downloads/vivienne/ML/Regression_UW/kc_house_test_data.csv'    
train = pd.read_csv(trainPath, dtype=dtype_dict)
test = pd.read_csv(testPath, dtype=dtype_dict) 

features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
            'waterfront', 'view', 'condition', 'grade', 'sqft_above', 
            'sqft_basement', 'yr_built', 'yr_renovated']

#Create a normalized feature matrix from the TRAINING data
(train_features, train_output) = get_data(train, features, 'price')
norm_train_features, train_norms = normalize_features(train_features)
#Learn weights with different values of l1_penalty and tolerance
#Save results in dfs
train_initial_weights = np.zeros(len(features))
weights1e7 = lasso_cyclical_coordinate_descent(norm_train_features, 
                                               train_output, 
                                               train_initial_weights, 
                                               l1_penalty, tolerance)
result1e7 = pd.DataFrame(weights1e7, index = features)

l1_penalty=1e8
weights1e8 = lasso_cyclical_coordinate_descent(norm_train_features, 
                                               train_output, 
                                               train_initial_weights, 
                                               l1_penalty, tolerance)
result1e8 = pd.DataFrame(weights1e8, index = features)

l1_penalty=1e4
tolerance=5e5
weights1e4 = lasso_cyclical_coordinate_descent(norm_train_features, 
                                               train_output, 
                                               train_initial_weights, 
                                               l1_penalty, tolerance)
result1e4 = pd.DataFrame(weights1e4, index = features)

'''
Recall that we normalized our feature matrix, before learning the weights. 
To use these weights on a test set, 
we must normalize the test data in the same way.
Alternatively, we can rescale the learned weights to 
include the normalization, so we never have to worry about normalizing
the test data:
'''    
#Compute the weights for the original features by 
#performing element-wise division --> Rescale learned weights
weights1e7_normalized = weights1e7 / train_norms
weights1e8_normalized = weights1e8 / train_norms
weights1e4_normalized = weights1e4 / train_norms
print (weights1e7_normalized[3]) #should print 161.31745624837794    

#Evaluate each of the learned models on the test data
(test_features, test_output) = get_data(test, features, 'price')
RSS_test = np.zeros(3)
weights_norm = np.array([weights1e7_normalized, weights1e8_normalized, 
                         weights1e4_normalized])
for i in range(3):
    RSS_test[i] = ((predict_outcome(test_features, weights_norm[i]) - 
                    test_output)**2).sum()     

print(RSS_test) #[  2.75962076e+14   5.37166151e+14   2.28459959e+14]