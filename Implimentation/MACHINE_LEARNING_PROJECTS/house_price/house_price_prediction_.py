# -*- coding: utf-8 -*-
"""House Price Prediction .ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1oM3ef2IBxxi-uFvm5_apaxUF-t0LJIxG
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

from sklearn.datasets import fetch_california_housing
house_price_dataset = fetch_california_housing()
print(house_price_dataset)

#loading the data set to the pandas data frame
house_price_dataframe=pd.DataFrame(house_price_dataset.data)
#print first 5 row of the dataframe
house_price_dataframe.head()

#loading the data set to the pandas data frame
house_price_dataframe=pd.DataFrame(house_price_dataset.data,columns=house_price_dataset.feature_names)
#print first 5 row of the dataframe
house_price_dataframe.head()

#add the target column to the data set
house_price_dataframe['Price']=house_price_dataset.target
house_price_dataframe.head()

#checking the no of row and columns in the data set
house_price_dataframe.shape

#check for missing values
house_price_dataframe.isnull().sum()

# statistical measures of the dataset
house_price_dataframe.describe()

coorelation=house_price_dataframe.corr()
#constructing the heatmap to understand the coorelation
plt.figure(figsize=(10,10))
sns.heatmap(coorelation,cbar=True,square=True,fmt='.2f',annot=True,annot_kws={'size':8},cmap='Blues')

#splitting the data and target
x=house_price_dataframe.drop(['Price'],axis=1)
y=house_price_dataframe['Price']
print(x)
print(y)

#spliting the data (test and train)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
print(x.shape,x_train.shape,x_test.shape)

#MODEL TRAINING
#XGBOOST REGRESSOR
# LOADING THE MODEL
model=XGBRegressor()
# TRIANING THE MODEL WITH X_TRAIN
model.fit(x_train,y_train)

#EVALUATION
#here we camnot find directly the accuracy because here we have to predict the value
# we will use error in the prediction and others
train_data_prediction=model.predict(x_train)
print(train_data_prediction)

#now predict the error form the y_train
#R SQUARE ERROR
score_1=metrics.r2_score(y_train,train_data_prediction)
print('R square error :',score_1)
score2=metrics.mean_absolute_error(y_train,train_data_prediction)
print('Mean absolute error :',score2)

plt.scatter(y_train,train_data_prediction)
plt.xlabel('Actual price')
plt.ylabel('Predicted price')
plt.title('Actual price vs Predicted price')
plt.show()

#predicting the test data
test_data_prediction = model.predict(x_test)

# Calculate R-squared error
score_3 = metrics.r2_score(y_test, test_data_prediction)
print('R square error:', score_3)

# Calculate Mean Absolute Error (MAE)
score_4 = metrics.mean_absolute_error(y_test, test_data_prediction)
print('Mean absolute error:', score_4)