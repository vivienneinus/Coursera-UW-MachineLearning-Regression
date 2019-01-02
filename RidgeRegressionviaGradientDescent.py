# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 09:31:21 2018

@author: vwzheng
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import os
import matplotlib.pyplot as plt
#%matplotlib inline
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 
              'sqft_living15':float, 'grade':int, 'yr_renovated':int, 
              'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 
              'sqft_lot15':float, 'sqft_living':float, 'floors':float, 
              'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 
              'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
path=("D:/Downloads/vivienne/ML/Regression_UW/Wk4_RidgeRegression")
os.chdir(path)
file='D:/Downloads/vivienne/ML/Regression_UW/kc_house_data.csv'
sales = pd.read_csv(file, dtype=dtype_dict)

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
    return(feature_matrix, output_array)

#predicted output is dot product between features matrix and weights     
def predict_outcome(feature_matrix, weights):
    predictions = np.dot(feature_matrix, weights) 
    return(predictions)

'''
Cost(w) = SUM[ (prediction - output)^2 ] + 
          l2_penalty*(w[0]^2 + w[1]^2 + ... + w[k]^2)
'''
#Compute the derivative of the regression cost function
def feature_derivative_ridge(errors, feature, weight, l2_penalty, 
                             feature_is_constant):
    if feature_is_constant:
        #derivative of the RSS with respect to w[i] 
        #=2*SUM[ error*[feature_i] ]
        derivative = 2*np.dot(errors, feature)
    else:
        #derivative of the regularization term with respect to w[i]
        #2*l2_penalty*w[i]
        #sum of both
        #=2*SUM[ error*[feature_i] ] + 2*l2_penalty*w[i]
        derivative = 2*np.dot(errors, feature) + 2*(l2_penalty*weight)
    return derivative

(example_features, example_output) = get_data(sales, ['sqft_living'], 
                                              'price')
my_weights = np.array([1., 10.])
test_predictions = predict_outcome(example_features, my_weights)
errors = test_predictions - example_output # prediction errors

# next two lines should print the same values
print(feature_derivative_ridge(errors, example_features.iloc[:,1], 
                               my_weights[1], 1, False))
print(np.sum(errors*example_features.iloc[:,1])*2+20.)
print('')

# next two lines should print the same values
print(feature_derivative_ridge(errors, example_features.iloc[:,0], 
                               my_weights[0], 1, True))
print(np.sum(errors)*2.)

'''
write a gradient descent function using your derivative function above. 
For each step in the gradient descent, we update the weight for each feature 
before computing our stopping criteria.
'''
def ridge_regression_gradient_descent(feature_matrix,output, initial_weights,
                                      step_size, l2_penalty,
                                      max_iterations=100):
    weights = np.array(initial_weights) # make sure it's a numpy array
    #while not reached maximum number of iterations:
    while max_iterations > 0:
        # compute the predictions using your predict_outcome() function
        predictions = predict_outcome(feature_matrix, weights)
        # compute the errors as predictions - output
        errors = predictions - output
        for i in range(len(weights)): # loop over each weight
            '''
            feature_matrix[:,i] is the feature column associated with
            weights[i], compute the derivative for weight[i];
            when i=0, compute the derivative of constant
            '''
            if i == 0 :
                feature_is_constant = True
            else:
                feature_is_constant = False
            derivative = feature_derivative_ridge(errors, 
                                                  feature_matrix.iloc[:,i], 
                                                  weights[i], l2_penalty, 
                                                  feature_is_constant)
            #subtract the step size times the derivative from current weight
            weights[i] -= step_size * derivative    
        max_iterations -= 1    
    return weights

#def(simple_feature)
simple_feature = ['sqft_living']
my_output = 'price'
#Split the dataset into training set and test set
train_data, test_data = train_test_split(sales, test_size = .2)
#Convert the training set and test set 
(feature_matrix, output) = get_data(train_data, simple_feature, my_output)
(test_feature_matrix, test_output) = get_data(test_data, simple_feature, 
                                              my_output)

step_size = 1e-12
max_iterations = 1000
init_s_weights = [0., 0.]
#Set the L2 penalty to 0.0 and run your ridge regression algorithm to 
#learn the weights of the simple model (described above)
simple_weights_0_penalty = ridge_regression_gradient_descent(feature_matrix,
                                                             output, 
                                                             init_s_weights,
                                                             step_size, 
                                                             0.0, 
                                                             max_iterations)
#Set the L2 penalty to high as 1e11 and run your ridge regression to learn 
#the weights of the simple model. Use the same parameters as above
simple_weights_h_penalty = ridge_regression_gradient_descent(feature_matrix,
                                                             output, 
                                                             init_s_weights,
                                                             step_size, 
                                                             1e11, 
                                                             max_iterations)

plt.plot(feature_matrix, output, 'k.',
         feature_matrix, predict_outcome(feature_matrix, 
                                         simple_weights_0_penalty), 'b-',
         feature_matrix, predict_outcome(feature_matrix, 
                                         simple_weights_h_penalty), 'r-')
#simple model RSS's for test set with initial weights of 0's, 
#weights of 0 penalty and weights of high penalty        
RSS_Test_simple_init = ((predict_outcome(test_feature_matrix, 
                                         init_s_weights) - 
                         test_output)**2).sum()              
RSS_Test_simple_0_pen = ((predict_outcome(test_feature_matrix, 
                                          simple_weights_0_penalty) - 
                          test_output)**2).sum()
RSS_Test_simple_high_pen = ((predict_outcome(test_feature_matrix, 
                                             simple_weights_h_penalty) - 
                             test_output)**2).sum()              

#a model with 2 features
model_features = ['sqft_living', 'sqft_living15']
my_output = 'price'
(features_matrix, output) = get_data(train_data, model_features, my_output)
(test_features_matrix, test_output) = get_data(test_data, model_features, 
                                               my_output)
init_m_weights = [0., 0., 0.]
#Set the L2 penalty to 0.0 and run your ridge regression algorithm to 
#learn the weights of the multiple model (described above)
multiple_weights_0_penalty = ridge_regression_gradient_descent(features_matrix,
                                                               output, 
                                                               init_m_weights,
                                                               step_size, 
                                                               0.0, 
                                                               max_iterations)
#Set the L2 penalty to high as 1e11 and run your ridge regression to learn 
#the weights of the multiple model. Use the same parameters as above
multiple_weights_h_penalty = ridge_regression_gradient_descent(features_matrix,
                                                               output, 
                                                               init_m_weights,
                                                               step_size, 
                                                               1e11, 
                                                               max_iterations)

plt.plot(features_matrix, output, 'k.',
         features_matrix, predict_outcome(features_matrix, 
                                          multiple_weights_0_penalty), 'b-',
         features_matrix, predict_outcome(features_matrix, 
                                          multiple_weights_h_penalty), 'r-')
         
#multiple model RSS's for test set with initial weights of 0's, 
#weights of 0 penalty and weights of high penalty        
RSS_Test_multiple_init = ((predict_outcome(test_features_matrix, 
                                           init_m_weights) - 
                           test_output)**2).sum()              
RSS_Test_multiple_0_pen = ((predict_outcome(test_features_matrix, 
                                            multiple_weights_0_penalty) - 
                            test_output)**2).sum()
RSS_Test_multiple_high_pen = ((predict_outcome(test_features_matrix, 
                                               multiple_weights_h_penalty) - 
                               test_output)**2).sum()                     
 
trainPath = 'D:/Downloads/vivienne/Regression_UW/kc_house_train_data.csv'
testPath = 'D:/Downloads/vivienne/Regression_UW/kc_house_test_data.csv'    
train = pd.read_csv(trainPath, dtype=dtype_dict)
test = pd.read_csv(testPath, dtype=dtype_dict)         

simple_feature = ['sqft_living']
my_output = 'price'
#Convert the training set and test set 
(fm, op) = get_data(train, simple_feature, my_output)
(test_fm, test_op) = get_data(test, simple_feature, my_output)

step_size = 1e-12
max_iterations = 1000
init_s_weights = [0., 0.]
#Set the L2 penalty to 0.0 and run your ridge regression algorithm to 
#learn the weights of the simple model (described above)
s_w_0_penalty = ridge_regression_gradient_descent(fm, op, init_s_weights,
                                                  step_size, 0.0, 
                                                  max_iterations)
#Set the L2 penalty to high as 1e11 and run your ridge regression to learn 
#the weights of the simple model. Use the same parameters as above
s_w_h_penalty = ridge_regression_gradient_descent(fm, op, init_s_weights,
                                                  step_size, 1e11, 
                                                  max_iterations)

plt.plot(fm, op, 'k.',
         fm, predict_outcome(fm, s_w_0_penalty), 'b-',
         fm, predict_outcome(fm, s_w_h_penalty), 'r-')
#simple model RSS's for test set with initial weights of 0's, 
#weights of 0 penalty and weights of high penalty        
RSS_T_s_init = ((predict_outcome(test_fm, init_s_weights)-test_op)**2).sum()              
RSS_T_s_0_pen = ((predict_outcome(test_fm, s_w_0_penalty)-test_op)**2).sum()
RSS_T_s_h_pen = ((predict_outcome(test_fm, s_w_h_penalty)-test_op)**2).sum()              

#a model with 2 features
model_features = ['sqft_living', 'sqft_living15']
my_output = 'price'
(fsm, op) = get_data(train, model_features, my_output)
(test_fsm, test_op) = get_data(test, model_features, my_output)
init_m_weights = [0., 0., 0.]
#Set the L2 penalty to 0.0 and run your ridge regression algorithm to 
#learn the weights of the multiple model (described above)
m_w_0_penalty = ridge_regression_gradient_descent(fsm, op, init_m_weights,
                                                  step_size, 0.0, 
                                                  max_iterations)
#Set the L2 penalty to high as 1e11 and run your ridge regression to learn 
#the weights of the multiple model. Use the same parameters as above
m_w_h_penalty = ridge_regression_gradient_descent(fsm, op, init_m_weights,
                                                  step_size, 1e11, 
                                                  max_iterations)

plt.plot(fsm, op, 'k.',
         fsm, predict_outcome(fsm, m_w_0_penalty), 'b-',
         fsm, predict_outcome(fsm, m_w_h_penalty), 'r-')
         
#multiple model RSS's for test set with initial weights of 0's, 
#weights of 0 penalty and weights of high penalty        
RSS_T_m_init = ((predict_outcome(test_fsm, init_m_weights)-test_op)**2).sum()              
RSS_T_m_0_pen = ((predict_outcome(test_fsm, m_w_0_penalty)-test_op)**2).sum()
RSS_T_m_h_pen = ((predict_outcome(test_fsm, m_w_h_penalty)-test_op)**2).sum()