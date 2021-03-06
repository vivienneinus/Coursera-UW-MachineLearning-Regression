# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 09:56:37 2018

@author: vwzheng
"""

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 
              'sqft_living15':float, 'grade':int, 'yr_renovated':int, 
              'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 
              'sqft_lot15':float, 'sqft_living':float, 'floors':str, 
              'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 
              'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

def polynomial_dataframe(feature, degree): # feature is pandas.Series type
    # assume that degree >= 1
    # initialize the dataframe:
    poly_dataframe = pd.DataFrame()
    # and set poly_dataframe['power_1'] equal to the passed feature
    poly_dataframe['power_1'] = feature
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        for power in range(2, degree+1):
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # assign poly_dataframe[name] to be feature^power; use apply(*)
            poly_dataframe[name] = feature.apply(lambda x: x**power)
    return poly_dataframe

path='D:/Downloads/vivienne/ML/Regression_UW/kc_house_data.csv'
sales = pd.read_csv(path, dtype=dtype_dict)
sales = sales.sort_values(['sqft_living', 'price'], ascending= [True, True])


#Make a 1 degree polynomial df with sales[‘sqft_living’] as the the feature 
poly1_data = polynomial_dataframe(sales['sqft_living'], 1)
poly1_data['price'] = sales['price']
#model1 = graphlab.linear_regression.create(poly1_data, target = 'price', 
#                                           features = ['power_1'], 
#                                           validation_set = None)
lm = LinearRegression(fit_intercept = True) 
#sklearn LR is for multiple LR model rather than simple LR
#newaxis can be used in all slicing operations to create an axis of length 1
#newaxis is an alias for ‘None’, used in place of this with the same result
model1 = lm.fit(poly1_data['power_1'][:, np.newaxis], poly1_data['price'])
#produce a scatter plot of the training data (just square feet vs price) 
#and add the fitted model
plt.plot(poly1_data['power_1'],poly1_data['price'],'.',
         poly1_data['power_1'], 
         model1.predict(poly1_data['power_1'][:, np.newaxis]),'-')

#Make a 2nd degree polynomial df with sales[‘sqft_living’] as the the feature 
poly2_data = polynomial_dataframe(sales['sqft_living'], 2)
poly2_data['price'] = sales['price']
model2 = lm.fit(poly2_data.iloc[:, poly2_data.columns != 'price'], 
                poly2_data['price'])
plt.plot(poly2_data['power_1'],poly2_data['price'],'.',
         poly2_data['power_1'], 
         model2.predict(poly2_data[['power_1', 'power_2']]),'-')

#Make a 3rd degree polynomial df with sales[‘sqft_living’] as the feature 
poly3_data = polynomial_dataframe(sales['sqft_living'], 3)
poly3_data['price'] = sales['price']
model3 = lm.fit(poly3_data.iloc[:, poly3_data.columns != 'price'], 
                poly3_data['price'])
plt.plot(poly3_data['power_1'],poly3_data['price'],'.',
         poly3_data['power_1'], 
         model3.predict(poly3_data[['power_1', 'power_2', 'power_3']]),'-')
#plt.show() #separate graphs
#Make a 15th degree polynomial df with sales[‘sqft_living’] as the feature
poly15_data = polynomial_dataframe(sales['sqft_living'], 15)
poly15_data['price'] = sales['price']
model15 = lm.fit(poly15_data.iloc[:, poly15_data.columns != 'price'], 
                poly15_data['price'])
plt.plot(poly15_data['power_1'],poly15_data['price'],'.',
         poly15_data['power_1'], 
         model15.predict(poly15_data.iloc[:, poly15_data.columns != 'price']),
         '-')
plt.show()

#sortBy column names, order: boolean with True for asc
def sort_read(fileName, sortBy, order):
    df = pd.read_csv(fileName, dtype = dtype_dict)
    df = df.sort_values(sortBy, ascending = order)
    return df

def polynomial_model(fileName, featureName, responseName, power):
    data = sort_read(fileName, [featureName, responseName], [True, True])
    poly = polynomial_dataframe(data[featureName], power)
    poly['price'] = data[responseName]
    model = lm.fit(poly.iloc[:, poly.columns != responseName], 
                   poly[responseName])
    plt.plot(poly['power_1'],poly[responseName],'.',
             poly['power_1'], 
             model.predict(poly.iloc[:, poly.columns != responseName]),'-')
    plt.show()
    return model.coef_[-1]

model1_15 = polynomial_model('wk3_kc_house_set_1_data.csv', 'sqft_living', 
                             'price', 15) 
model2_15 = polynomial_model('wk3_kc_house_set_2_data.csv', 'sqft_living', 
                             'price', 15) 
model3_15 = polynomial_model('wk3_kc_house_set_3_data.csv', 'sqft_living', 
                             'price', 15)
model4_15 = polynomial_model('wk3_kc_house_set_4_data.csv', 'sqft_living', 
                             'price', 15) 

#train_data = pd.read_csv('wk3_kc_house_train_data.csv', dtype = dtype_dict)
#test_data = pd.read_csv('wk3_kc_house_test_data.csv', dtype = dtype_dict)    
#valid_data = pd.read_csv('wk3_kc_house_valid_data.csv', dtype = dtype_dict) 
trainAndValid, test = train_test_split(sales, test_size = .1) 
train, valid = train_test_split(trainAndValid, test_size = .5)

def polynomial_RSS(df, power, featureName, responseName, df2):
    poly = polynomial_dataframe(df[featureName], power)
    model = lm.fit(poly, df[responseName])
    valid = polynomial_dataframe(df2[featureName], power)
    RSS = ((model.predict(valid) - df2[responseName])**2).sum()
    return RSS

def polynomials(df1, df2, highestPower):
    listRSS = []
    #special case of 1-power
    model = lm.fit(df1['sqft_living'][:, np.newaxis], df1['price'])
    RSS = ((model.predict(df2['sqft_living'][:, np.newaxis])
            -df2['price'])**2).sum()
    listRSS.append(RSS)
    for power in range(2, highestPower+1):
        RSS = polynomial_RSS(df1, power, 'sqft_living', 'price', df2)
        listRSS.append(RSS) 
    return listRSS

RSS_valid_15 = polynomials(train, valid, 15) 
power_selection = RSS_valid_15.index(min(RSS_valid_15)) + 1
RSS_test = polynomial_RSS(train, power_selection, 'sqft_living', 'price', 
                          test)
RSS_test_15 = polynomials(train, test, 15)
powerSelectedForTest = RSS_test_15.index(min(RSS_test_15)) + 1
 


        