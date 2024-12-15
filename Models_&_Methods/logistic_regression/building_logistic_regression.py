# -*- coding: utf-8 -*-
"""Building Logistic Regression.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1LT26AHXCKRCWVymFIBCM3WVPKVK6Td2_
"""

import numpy as np
#Step 1: Initialization

#Set Learning Rate: This determines the step size during parameter updates. A smaller learning rate leads to more precise adjustments, while a larger one can speed up convergence.
#Set Number of Iterations: This specifies the maximum number of times the model will go through the entire training dataset.
#Initialize Weights and Bias: Assign random values to these parameters, which will be gradually refined during training.
#Step 2: Build Logistic Regression Function

#Sigmoid Function: This function maps any real value to a value between 0 and 1. In logistic regression, it's used to calculate the probability of a data point belonging to a particular class.
#Step 3: Gradient Descent

#Update Parameters: Calculate the gradient of the cost function with respect to the weights and bias. This gradient indicates the direction of steepest ascent.
#Adjust Weights and Bias: Update the weights and bias in the opposite direction of the gradient, using the learning rate to control the step size.
#Step 4: Cost Function Minimization

#Iterative Process: Repeat steps 2 and 3 until the cost function reaches a minimum or the maximum number of iterations is reached.
#Best Model: The model with the lowest cost function is considered the best model.
#Step 5: Prediction

#Threshold: Set a threshold value (e.g., 0.5).
#Classification: If the predicted probability is greater than the threshold, classify the data point as belonging to one class; otherwise, classify it as belonging to the other class.

class logistic_regression():

  #initaiting alpha and iteration
  def __init__(self,alpha,iteration):
    self.alpha=alpha
    self.iteration=iteration


  def fit(self,x,y):
    self.m,self.n=x.shape
    self.w=np.zeros(self.n)
    self.b=0
    self.x=x
    self.y=y
    for i in range(self.iteration):
       self.update_weight()
  def update_weight(self):
    y_hat=1/(1+np.exp(-(self.x.dot(self.w)+self.b)))
    dw=(1/self.m)*np.dot(self.x.T,(y_hat-self.y))#we are taking transpose here because no of column must have same no of rows of the second
    db=(1/self.m)*np.sum(y_hat-self.y)
    self.w=self.w-self.alpha*dw
    self.b=self.b-self.alpha*db
  def predict(self,x):
    y_pred=1/(1+np.exp(-(x.dot(self.w)+self.b)))
    y_pred=np.where(y_pred>0.5,1,0)
    return y_pred

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

dataset=pd.read_csv('/content/diabetes.csv')
dataset.head()

dataset.shape
dataset.describe()

dataset['Outcome'].value_counts()
dataset.groupby('Outcome').mean()

x=dataset.drop(columns='Outcome',axis=1)
y=dataset['Outcome']
print(x)
print(y)

scaler=StandardScaler()
scaler.fit(x)
standardized_data=scaler.transform(x)
print(standardized_data)
print(x)
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
print(x.shape,x_train.shape,x_test.shape)

classifier=logistic_regression(alpha=0.1,iteration=1000)
classifier.fit(x_train,y_train)

x_train_pred=classifier.predict(x_train)
training_data_accuracy=accuracy_score(y_train,x_train_pred)
print(training_data_accuracy)
