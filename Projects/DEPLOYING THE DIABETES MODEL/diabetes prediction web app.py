# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:13:55 2024

@author: vedika patil
"""

import numpy as np
import pickle 
import streamlit as st



# Correctly formatted file path
file_path = "C:/Users/vedika patil/Desktop/CODING/AI ML/AI ML LEARNING/Projects/DEPLOYING THE DIABETES MODEL/trained_model.sav"

# Loading the model
loaded_model = pickle.load(open(file_path, 'rb'))  # read in binary mode



#creating a function
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    
  
    
def main():
    
    
    #giving a title
    st.title('Diabetes Prediction Web App')
    
    #getting the input data from the user
    
    Pregnancies=st.text_input('Number of Pregnencies')
    Glucose=st.text_input('Glucose level')
    BloodPressure=st.text_input('Blood Pressure Value')
    SkinThickness=st.text_input('SkinThickness Value')
    Insulin=st.text_input('Insulin Level')
    BMI=st.text_input('BMI Value')
    DiabetesPedigreeFunction=st.text_input('DiabetesPedigreeFunction Value')
    Age=st.text_input('Age Of The Person')
    
    
    
    #code for prediction
    diagnosis=''
    
    
    #creating a button for prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    st.success(diagnosis)
    
    
if __name__ =='__main__':
    main()