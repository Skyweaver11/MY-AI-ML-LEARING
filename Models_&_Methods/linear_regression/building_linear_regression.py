# -*- coding: utf-8 -*-
"""Building Linear Regression.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1jyj2OS1a00DrcK_iKMBeH51KaGn5gh34
"""

import numpy as np
#Work flow of the Linear Regression model:

#Step 1: Set Learning Rate & Number of Iterations; Initiate Random weight and bias value.
#Step 2: Build Linear Regression Equation. (y = wx + b)
#Step 3: Find the "y pred" value for given x value for the corresponding weight & bias.
#Step 4: Check the loss function for these parameter values. (difference between "y pred" & "true y")
#Step 5: Update the parameter values using Gradient Descent. (new weight & bias value)
#Step 6: Step 3, 4, 5 are repeated till we get minimum loss function.

#Finally we will get the best model (best weight and bias value) as it has minimum loss function.

class Linear_Regression():
#we define self because when we call this func into other variable then that variable would be replaced in place of self
  def __init__(self, learning_rate,no_of_iterations):
    self.learning_rate=learning_rate
    self.no_of_iterations=no_of_iterations

  def fit(self,x,y):#x_train and y_train
    #number of training examples and numbers
    self.m,self.n=x.shape       #number of rows and columns
    #initialsing the weight and bias
    self.w=np.zeros(self.n)
    self.b=0
    self.x=x
    self.y=y
    #impelmenting gradient descent
    for i in range (self.no_of_iterations):
      self.update_weights()

  def update_weights(self):
    y_prediction=self.predict(self.x)
    #calculating gradient
    dw=-(2*(self.x.T).dot(self.y-y_prediction))//self.m
    db=-2*np.sum(self.y-y_prediction)/self.m
    #updating the weight
    self.w=self.w-self.learning_rate*dw
    self.b=self.b-self.learning_rate*db

  def predict(self,x):
     return x.dot(self.w)+self.b

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

salary_data=pd.read_csv('/content/salary_data.csv')
salary_data.head()

salary_data.shape

salary_data.isnull().sum()

x=salary_data.iloc[:,:-1].values
y=salary_data.iloc[:,1].values
print(x)
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=2)

model=Linear_Regression(learning_rate=0.02,no_of_iterations=20000)

model.fit(x_train,y_train)

print('weight=',model.w[0])
print('bias=',model.b)

test_data_prediction=model.predict(x_test)

plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,test_data_prediction,color='blue')
plt.xlabel('work experience')
plt.ylabel('salary')
plt.title('salary vds exp')


