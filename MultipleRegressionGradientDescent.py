# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 16:50:46 2018

@author: vwzheng
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
file_train = 'D:/Downloads/vivienne/ML/Regression_UW/kc_house_train_data.csv'
train_data = pd.read_csv(file_train,
                         dtype = {'bathrooms':float, 'waterfront':int, 
                                  'sqft_above':int, 'sqft_living15':float, 
                                  'grade':int, 'yr_renovated':int, 
                                  'price':float, 'bedrooms':float, 
                                  'zipcode':str, 'long':float, 
                                  'sqft_lot15':float, 'sqft_living':float, 
                                  'floors':str, 'condition':int, 'lat':float, 
                                  'date':str, 'sqft_basement':int, 
                                  'yr_built':int, 'id':str, 'sqft_lot':int,
                                  'view':int})
df = pd.DataFrame(train_data)

#a list of features to be used as inputs, and a name of the output
def get_data(data, features, output):
    # add a constant column to a data_Frame
    data['constant'] = 1 
    # prepend variable 'constant' to the features list
    features.insert(0, 'constant')
    # select the columns of data_Frame given by the ‘features’ list
    feature_matrix = data[features]
    # assign the column of data associated with the target to the variable 
    #‘output’
    output_array = data[output]
    return(feature_matrix, output_array)
    
#predicted output is dot product between features matrix and weights     
def predict_outcome(feature_matrix, weights):
    predictions = np.dot(feature_matrix, weights) 
    return(predictions)

#derivative of regression cost function with respect to the weight of 
#‘feature’
#twice the dot product between ‘feature’ and ‘errors’    
def feature_derivative(errors, feature):
    derivative = 2*np.dot(errors, feature)
    return(derivative)

def regression_gradient_descent(feature_matrix, output, initial_weights, 
                                step_size, tolerance):
    converged = False
    weights = np.array(initial_weights)
    while not converged:
        # compute the predictions based on feature_matrix and weights:
        predictions = predict_outcome(feature_matrix, weights)
        # compute the errors as predictions - output:
        errors = predictions - output
        gradient_sum_squares = 0 # initialize the gradient
        # while not converged, update each weight individually:
        for i in range(len(weights)):
            # Recall that feature_matrix[:, i] is the feature column 
            #associated with weights[i]
            # compute the derivative for weight[i]:
            derivative = feature_derivative(errors, 
                                            feature_matrix.iloc[:,[i]])
            # add the squared derivative to the gradient magnitude
            gradient_sum_squares += derivative**2
            # update the weight based on step size and derivative:
            weights[i] = weights[i] - step_size*derivative
        gradient_magnitude = np.sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True
    return(weights)

simple_feature = ['sqft_living']
my_output= 'price'
(simple_feature_matrix, output) = get_data(df, simple_feature, my_output)
initial_weights = np.array([-47000., 1.])
step_size = 7e-12
tolerance = 2.5e7

simple_weights = regression_gradient_descent(simple_feature_matrix, output,
                                             initial_weights, step_size, 
                                             tolerance)    

model_features = ['sqft_living', 'sqft_living15']
my_output = 'price'
(model_feature_matrix, output) = get_data(df, model_features,my_output)
initial_weights = np.array([-100000., 1., 1.])
step_size = 4e-12
tolerance = 1e9
model_weights = regression_gradient_descent(model_feature_matrix, output, 
                                            initial_weights, step_size, 
                                            tolerance)
file_test = 'D:/Downloads/vivienne/Regression_UW/kc_house_test_data.csv'
test_data = pd.read_csv(file_test,
                        dtype = {'bathrooms':float, 'waterfront':int, 
                                  'sqft_above':int, 'sqft_living15':float, 
                                  'grade':int, 'yr_renovated':int, 
                                  'price':float, 'bedrooms':float, 
                                  'zipcode':str, 'long':float, 
                                  'sqft_lot15':float, 'sqft_living':float, 
                                  'floors':str, 'condition':int, 'lat':float, 
                                  'date':str, 'sqft_basement':int, 
                                  'yr_built':int, 'id':str, 'sqft_lot':int,
                                  'view':int})
test = pd.DataFrame(test_data)
test['constant'] = 1
def get_residual_sum_of_squares(input_features, output, weights):
    RS = (output - np.dot(input_features, weights))**2
    return(sum(RS))    
RSS_model = get_residual_sum_of_squares(test[model_features], 
                                        test[my_output],
                                        model_weights)
RSS_simple = get_residual_sum_of_squares(test[simple_feature], 
                                         test[my_output],
                                         simple_weights)    

#multiple regression
df['bed_bath_rooms'] = df['bedrooms']*df['bathrooms']
df['bedrooms_squared'] = df['bedrooms']**2
df['log_sqft_living'] = np.log(df['sqft_living'])
df['lat_plus_long'] = df['lat']+df['long']
test['bed_bath_rooms'] = test['bedrooms']*test['bathrooms']
test['bedrooms_squared'] = test['bedrooms']**2
test['log_sqft_living'] = np.log(test['sqft_living'])
test['lat_plus_long'] = test['lat']+test['long']
lm = linear_model.LinearRegression()
model1 = lm.fit(df[['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']], 
                df['price'])
coef_model1 = np.insert(model1.coef_, 0, model1.intercept_)
RSS_model1 = get_residual_sum_of_squares(df[['constant','sqft_living', 
                                             'bedrooms', 'bathrooms', 'lat', 
                                             'long']], df['price'], 
                                         coef_model1)
RSS_model1_test = get_residual_sum_of_squares(test[['constant','sqft_living', 
                                                   'bedrooms', 'bathrooms', 
                                                   'lat', 'long']], 
                                              test['price'], coef_model1)     
    
model2 = lm.fit(df[['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long', 
                    'bed_bath_rooms']], df['price'])
coef_model2 = np.insert(model2.coef_, 0, model2.intercept_)
RSS_model2 = get_residual_sum_of_squares(df[['constant','sqft_living', 
                                             'bedrooms', 'bathrooms', 'lat', 
                                             'long', 'bed_bath_rooms']], 
                                         df['price'], coef_model2)
RSS_model2_test = get_residual_sum_of_squares(test[['constant','sqft_living', 
                                                    'bedrooms', 'bathrooms', 
                                                    'lat', 'long', 
                                                    'bed_bath_rooms']], 
                                              test['price'], coef_model2)

model3 = lm.fit(df[['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long', 
                    'bedrooms_squared', 'log_sqft_living', 'lat_plus_long']], 
                df['price'])
coef_model3 = np.insert(model3.coef_, 0, model3.intercept_)
RSS_model3 = get_residual_sum_of_squares(df[['constant','sqft_living', 
                                             'bedrooms', 'bathrooms', 'lat', 
                                             'long', 'bedrooms_squared', 
                                             'log_sqft_living', 
                                             'lat_plus_long']], df['price'], 
                                         coef_model3)
RSS_model3_test = get_residual_sum_of_squares(test[['constant','sqft_living', 
                                                    'bedrooms', 'bathrooms', 
                                                    'lat', 'long', 
                                                    'bedrooms_squared', 
                                                    'log_sqft_living', 
                                                    'lat_plus_long']], 
                                              test['price'], coef_model3)