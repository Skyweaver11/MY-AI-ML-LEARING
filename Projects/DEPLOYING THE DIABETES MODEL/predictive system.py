# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

# Correctly formatted file path
file_path = "C:/Users/vedika patil/Desktop/CODING/AI ML/AI ML LEARNING/Projects/DEPLOYING THE DIABETES MODEL/trained_model.sav"

# Loading the model
loaded_model = pickle.load(open(file_path, 'rb'))  # read in binary mode

input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')