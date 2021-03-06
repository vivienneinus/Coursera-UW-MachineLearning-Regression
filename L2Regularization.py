# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 11:22:53 2018

@author: vwzheng
"""
#??why sorted train data has a larger RSS mean than unsorted for k-foldCV
import pandas as pd
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import os
path=('D:/Downloads/vivienne/ML/Regression_UW/Wk3_PolynomialRegression')
os.chdir(path)
from PolynomialRegression import polynomial_dataframe, sort_read

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 
              'sqft_living15':float, 'grade':int, 'yr_renovated':int, 
              'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 
              'sqft_lot15':float, 'sqft_living':float, 'floors':float, 
              'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 
              'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
#sort_read for plots
set_1 = sort_read('wk3_kc_house_set_1_data.csv', ['sqft_living', 'price'], 
                  [True, True])
set_2 = sort_read('wk3_kc_house_set_2_data.csv', ['sqft_living', 'price'], 
                  [True, True])
set_3 = sort_read('wk3_kc_house_set_3_data.csv', ['sqft_living', 'price'], 
                  [True, True])
set_4 = sort_read('wk3_kc_house_set_4_data.csv', ['sqft_living', 'price'], 
                  [True, True])
test = pd.read_csv('wk3_kc_house_test_data.csv', dtype = dtype_dict)

os.chdir("D:/Downloads/vivienne/ML/Regression_UW/Wk4_RidgeRegression")
filePath = 'D:/Downloads/vivienne/ML/Regression_UW/kc_house_data.csv'
sales = pd.read_csv(filePath, dtype=dtype_dict)
sales = sales.sort_values(['sqft_living', 'price'], ascending= [True, True])
#sales = sort_read(filePath, ['sqft_living', 'price'], [True, True])
l2_small_penalty = 1.5e-5

poly15_data = polynomial_dataframe(sales['sqft_living'], 15) 
model = linear_model.Ridge(alpha=l2_small_penalty, normalize=True)
model.fit(poly15_data, sales['price'])


os.chdir("D:/Downloads/vivienne/ML/Regression_UW/Wk4_RidgeRegression")
l2_small_penalty=1e-9

#ridge regression used to prevent overfitting
#features supposed to have more than 1D
def ridge_model(features, response, l2_penalty):
    model = linear_model.Ridge(alpha=l2_penalty, normalize=True)
    model.fit(features, response)
    return model

def polynomial_ridge(data, featureName, responseName, power, l2_penalty):
    poly = polynomial_dataframe(data[featureName], power)
    if power == 1:
        model = linear_model.Ridge(alpha=l2_penalty, normalize=True)
        model.fit(poly[:, np.newaxis], data[responseName])
        plt.plot(poly, data[responseName],'.',
                 poly, model.predict(poly[:, np.newaxis]),'-')
        plt.show()
    else:
        model = ridge_model(poly, data[responseName], l2_penalty)
        plt.plot(poly['power_1'],data[responseName],'.',
                 poly['power_1'], model.predict(poly),'-')
        plt.show()
    return model


model15_1 = polynomial_ridge(set_1, 'sqft_living', 'price', 15, 
                             l2_small_penalty)

model15_2 = polynomial_ridge(set_2, 'sqft_living', 'price', 15, 
                             l2_small_penalty)

model15_3 = polynomial_ridge(set_3, 'sqft_living', 'price', 15, 
                             l2_small_penalty)

model15_4 = polynomial_ridge(set_4, 'sqft_living', 'price', 15, 
                             l2_small_penalty)
print(model15_1.coef_[0], model15_2.coef_[0],
      model15_3.coef_[0], model15_4.coef_[0])

l2_large_penalty=1.23e2
model15_1_large = polynomial_ridge(set_1, 'sqft_living', 'price', 15, 
                                   l2_large_penalty)
model15_2_large = polynomial_ridge(set_2, 'sqft_living', 'price', 15, 
                                   l2_large_penalty)
model15_3_large = polynomial_ridge(set_3, 'sqft_living', 'price', 15, 
                                   l2_large_penalty)
model15_4_large = polynomial_ridge(set_4, 'sqft_living', 'price', 15, 
                                   l2_large_penalty)
print(model15_1_large.coef_[0], model15_2_large.coef_[0], 
      model15_3_large.coef_[0], model15_4_large.coef_[0])

#implement polynomial RSS calculation
def model_RSS(features1, features2, response1, response2, l2_penalty):
    model = ridge_model(features1, response1, l2_penalty)
    RSS = ((model.predict(features2) - response2)**2).sum()    
    return RSS

#implement k-fold cross-validation, return an average of RSS's from each CV
#divid the whole obs into k segments, segment i is the valid group    
#choose the remainder of the data that's not part of the segment i 
#combine two slices (0:start) and (end+1:n) together as the train      
def k_fold_cross_validation(k, l2_penalty, data, output):
    n = len(data)
    RSSs = []
    for i in range(k):
        start = int((n*i)/k)
        end = int((n*(i+1))/k-1)
        #print(start, end)
        valid = data[start:end+1]
        output_valid = output[start:end+1]
        train = data[0:start].append(data[end+1:n])
        output_train = output[0:start].append(output[end+1:n])
        RSSs.append(model_RSS(train, valid, output_train, output_valid,
                              l2_penalty))                      
    return np.mean(RSSs)

train_valid_shuffled = pd.read_csv('wk3_kc_house_train_valid_shuffled.csv', 
                                   dtype = dtype_dict)
#For each l2_penalty in [10^3, 10^3.5, 10^4, 10^4.5, ..., 10^9] 
#Run 10-fold cross-validation with l2_penalty
#Fit a 15th-order polynomial model using the sqft_living input    
data15 = polynomial_dataframe(train_valid_shuffled['sqft_living'], 15)
valid_errs = pd.DataFrame(columns=['l2_penalty', 'avgRSS'])
index = 0
#valid_errs = {}
for l2_penalty in np.logspace(3, 9, num=13):
    RSS_bar = k_fold_cross_validation(10, l2_penalty, data15,
                                      train_valid_shuffled['price'])
    #valid_errs.update({RSS_bar: l2_penalty})
    valid_errs.loc[index] = [l2_penalty, RSS_bar]
    index += 1

penaltyOpt = valid_errs[valid_errs['avgRSS']==min(valid_errs['avgRSS'])
                        ]['l2_penalty'] # this is a series 
#a series has an index embeded as a dataframe with one column
   
#l2_penalty_min = valid_errs.get(min(valid_errs.keys()))
#keys = np.fromiter(Samples.keys(), dtype=float)
#vals = np.fromiter(Samples.values(), dtype=float)
plt.plot(valid_errs['l2_penalty'], valid_errs['avgRSS'], 'k.')
plt.xscale('log')
#retrain a final model on all of the training data 
#using best value of l2_penalty
RSS_bar = k_fold_cross_validation(10, penaltyOpt[0], data15,
                                  train_valid_shuffled['price'])
test15 = polynomial_dataframe(test['sqft_living'], 15)
RSS_test = model_RSS(data15, test15, train_valid_shuffled['price'],
                     test['price'], penaltyOpt[0])

