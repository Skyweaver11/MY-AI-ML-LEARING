# -*- coding: utf-8 -*-
"""Diabetes Prediction using Machine Learning .ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13uEhrVuz0L3SFKSbIgH0eQ3g4wQTuvVV
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#data collection adn analysis
#loading the diabetes dataset to a pandas dataframe
diabetes_dataset = pd.read_csv('/diabetes.csv')

#if you do not know what does the function does then only add ? to the end
#pd.read_csv?

from google.colab import drive
drive.mount('/content/drive')

# read the starting 5 data
diabetes_dataset.head()

# to know the rows and column
diabetes_dataset.shape

# getting the sattistical measures of the data
diabetes_dataset.describe()

# 25% and 50% 75% means that the value is less the 25% if ans is 1 then all the values are less and if 0 the non is less then 25%

# diabetes count
diabetes_dataset['Outcome'].value_counts()

diabetes_dataset.groupby('Outcome').mean()

#to seperate the data and lables
X=diabetes_dataset.drop(columns='Outcome',axis=1)
Y=diabetes_dataset['Outcome']

print (X)
print (Y)

# data stadardization there is variety of ranges in the rows like age etc and if we put the data as it is then it will be hard to predict the output
# so we use standardization in perticular range
scaler= StandardScaler()

scaler.fit(X)
standardized_data=scaler.transform(X)
# instead of the above we can also use scaler.fit_transform

print (standardized_data)

X=standardized_data
Y=diabetes_dataset['Outcome']
print(X)
print(Y)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=2)
#statify ko include nhi kiya to sare diabetic test case may go to x_train and non diabetic may go to x_test
#no of diabetic and non diabetic patients remains same in x_train  and test also

print(X.shape,X_train.shape,X_test.shape)

#TRAINING THE MODEL
classifier=svm.SVC(kernel='linear')

#training the svm classifier
classifier.fit(X_train,Y_train)

#model evaluation
#FINDING THE ACCURACY SCORE
X_train_prediction=classifier.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print (training_data_accuracy)

X_test_prediction=classifier.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print (test_data_accuracy)

input_data=(0,137,40,35,168,43.1,2.288,33)
#change the list to numpy array
input_data_as_numpy_array=np.asarray(input_data)
#reshape the array as we are predicting for one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
#standardize the input data
std_data=scaler.transform(input_data_reshaped)
print(std_data)
prediction=classifier.predict(std_data)
print(prediction)
if (prediction[0]==0):
  print('the person is not diabetic')
else:
  print('the person is diabetic')