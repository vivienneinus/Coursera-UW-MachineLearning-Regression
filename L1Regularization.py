# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 12:09:52 2018

@author: vwzheng
"""
import os
import pandas as pd
import numpy as np 
from math import *

from sklearn import linear_model  # using scikit-learn

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 
              'sqft_living15':float, 'grade':int, 'yr_renovated':int, 
              'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 
              'sqft_lot15':float, 'sqft_living':float, 'floors':float, 
              'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 
              'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

filePath = 'D:/Downloads/vivienne/ML/Regression_UW/kc_house_data.csv'
sales = pd.read_csv(filePath, dtype=dtype_dict)
#Create new features
sales['sqft_living_sqrt'] = sales['sqft_living'].apply(sqrt)
sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(sqrt)
sales['bedrooms_square'] = sales['bedrooms']*sales['bedrooms']
sales['floors_square'] = sales['floors']*sales['floors']

all_features = ['bedrooms', 'bedrooms_square', 'bathrooms', 'sqft_living', 
                'sqft_living_sqrt', 'sqft_lot', 'sqft_lot_sqrt', 'floors', 
                'floors_square', 'waterfront', 'view', 'condition', 'grade',
                'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']

#Using an entire house dataset, learn regression weights with L1 penalty(5e2)
model_all = linear_model.Lasso(alpha=5e2, normalize=True) # set parameters
model_all.fit(sales[all_features], sales['price']) # learn weights
feature_select_all = pd.DataFrame({'features': all_features,
                                   'weights': model_all.coef_})

#To find a good L1 penalty, 
#we will explore multiple values using a validation set. 
os.chdir("D:/Downloads/vivienne/ML/Regression_UW/Wk3_PolynomialRegression")
testing = pd.read_csv('wk3_kc_house_test_data.csv', dtype=dtype_dict)
training = pd.read_csv('wk3_kc_house_train_data.csv', dtype=dtype_dict)
validation = pd.read_csv('wk3_kc_house_valid_data.csv', dtype=dtype_dict)

#Create new features for test, train and valid sets
testing['sqft_living_sqrt'] = testing['sqft_living'].apply(sqrt)
testing['sqft_lot_sqrt'] = testing['sqft_lot'].apply(sqrt)
testing['bedrooms_square'] = testing['bedrooms']*testing['bedrooms']
testing['floors_square'] = testing['floors']*testing['floors']

training['sqft_living_sqrt'] = training['sqft_living'].apply(sqrt)
training['sqft_lot_sqrt'] = training['sqft_lot'].apply(sqrt)
training['bedrooms_square'] = training['bedrooms']*training['bedrooms']
training['floors_square'] = training['floors']*training['floors']

validation['sqft_living_sqrt'] = validation['sqft_living'].apply(sqrt)
validation['sqft_lot_sqrt'] = validation['sqft_lot'].apply(sqrt)
validation['bedrooms_square'] = validation['bedrooms']*validation['bedrooms']
validation['floors_square'] = validation['floors']*validation['floors']

#Calculate RSS for linear_model.Lasso
def Lasso_RSS(features1, response1, features2, response2, l1_penalty):
    lml = linear_model.Lasso(alpha = l1_penalty, normalize = True)
    lml.fit(features1, response1)
    RSS = ((lml.predict(features2) - response2)**2).sum()
    return RSS

def Lasso_model(features, response, l1_penalty):
    model = linear_model.Lasso(alpha=l1_penalty, normalize=True)
    model.fit(features, response)
    return model

def model_RSS(features1, features2, response1, response2, l1_penalty):
    model = Lasso_model(features1, response1, l1_penalty)
    RSS = ((model.predict(features2) - response2)**2).sum()    
    return RSS

#Search for best value for l1 penalty with a for loop
#On validation data with lowest RSS
RSS_valid = pd.DataFrame(columns=['l1_penalty', 'RSS'])
index = 0
for l1_penalty in np.logspace(1, 7, num=13):
    #Learn a model on TRAINING data using the specified l1_penalty.
    '''
    lml = linear_model.Lasso(alpha=l1_penalty, normalize=True)
    lml.fit(training[all_features], training['price'])
    RSS=((lml.predict(validation[all_features])-validation['price'])**2
         ).sum()
    '''
    RSS=Lasso_RSS(training[all_features], training['price'],
                  validation[all_features], validation['price'], l1_penalty)
    RSS_valid.loc[index] = [l1_penalty, RSS]
    index += 1
    
#lowest RSS_valid = 3.982133e+14 with l1 penalty = 10    
l1_penalty_optimal = RSS_valid[RSS_valid['RSS']==min(RSS_valid['RSS'])
                               ]['l1_penalty'] #return a series    
lpo = np.array(l1_penalty_optimal) #numpy array of 1 float

'''
RSS_test = Lasso_RSS(training[all_features], training['price'], 
                     testing[all_features], testing['price'],
                     l1_penalty_optimal[0]) 
#l1_penalty_optimal[0] returns the numeric; otherwise python gets confused  
'''
RSS_test = Lasso_RSS(training[all_features], training['price'], 
                     testing[all_features], testing['price'], lpo)
#98467402552698.75

#Using the best L1 penalty, count nonzero weights
model_optimal = Lasso_model(training[all_features], training['price'], lpo)

nonzero_weights = np.count_nonzero(model_optimal.coef_) + np.count_nonzero(
                  model_optimal.intercept_)

#Search for best value for l1 penalty with a for loop
def searchForOptimal(x1, y1, x2, y2, penalties, str1, str2, str3): 
    RSS_valid = pd.DataFrame(columns=[str1, str2, str3])
    index = 0
    for l1_penalty in penalties:
        #Learn a model on TRAINING data using the specified l1_penalty.
        model = Lasso_model(x1, y1, l1_penalty)
        count_nonzero = np.count_nonzero(model.coef_) + np.count_nonzero(
                        model.intercept_)
        RSS = ((model.predict(x2) - y2)**2).sum()
        RSS_valid.loc[index] = [l1_penalty, RSS, count_nonzero]
        index += 1    
    
    return RSS_valid

'''
#Explore large range of l1_penalty
for l1_penalty_large in np.logspace(1, 4, num=20):
    #Learn a model on TRAINING data using the specified l1_penalty. 
    model = linear_model.Lasso(alpha=l1_penalty, normalize=True)
'''    
RSS_valid_large = searchForOptimal(training[all_features], training['price'], 
                                   validation[all_features], 
                                   validation['price'],
                                   np.logspace(1, 4, num=20), 'l1_penalty', 
                                   'RSS', 'count_nonzero')    
#lowest RSS on valid data with l1 penalty 
#and count of nonzero weights incl intercept    
l1_penalty_large_optimal=RSS_valid_large[RSS_valid_large['RSS']==min(
                                         RSS_valid_large['RSS'])]
#return a series    
lpol = np.array(l1_penalty_large_optimal.iloc[:,0])

max_nonzeros = 7

#lower bound for l1 penalty; otherwise, too many features selected
l1_penalty_min=RSS_valid_large[RSS_valid_large['count_nonzero']>max_nonzeros
                               ]['l1_penalty'].max()

#upper bound for l1 penalty; otherwise, too few features selected
l1_penalty_max=RSS_valid_large[RSS_valid_large['count_nonzero']<max_nonzeros
                               ]['l1_penalty'].min()

#Explore narrow range of l1 penalties
RSS_valid_narrow =searchForOptimal(training[all_features], training['price'], 
                                   validation[all_features], 
                                   validation['price'],
                                   np.linspace(l1_penalty_min, 
                                               l1_penalty_max, 20), 
                                   'l1_penalty', 'RSS', 'count_nonzero')
                                   
#lowest RSS on VALIDATION set and has sparsity equal to ‘max_nonzeros’  
l1_penalty_narrow_optimal =RSS_valid_narrow[RSS_valid_narrow['RSS']==
                                            RSS_valid_narrow[
                                            RSS_valid_narrow['count_nonzero'] 
                                            == max_nonzeros]['RSS'].min()]  
#best value for l1 penalty within the narrow range
lpon = np.array(l1_penalty_narrow_optimal.iloc[:,0])     
##Using train dataset, learn regression weights with the best L1 penalty
model_lpon = Lasso_model(training[all_features], training['price'], lpon)
#Return nonzero weights with selected features
feature_select = pd.DataFrame({'features': all_features,
                               'weights': model_lpon.coef_})
#feature_select = pd.DataFrame(columns=['features', 'weights'])    
#feature_select['features'] = all_features
#feature_select['weights'] = model_lpon.coef_

                            