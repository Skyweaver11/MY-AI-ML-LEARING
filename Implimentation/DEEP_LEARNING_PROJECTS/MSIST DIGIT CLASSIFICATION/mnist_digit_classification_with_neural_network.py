# -*- coding: utf-8 -*-
"""MNIST Digit Classification with Neural Network.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1sTTIpGgaElkCsnvlPRgGe3MMJ7WpkHYT
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from google.colab.patches import cv2_imshow
from PIL import Image
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras
from keras.datasets import mnist
from tensorflow.math import confusion_matrix

(x_train, y_train), (x_test, y_test) = mnist.load_data()

type(x_train)

print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

print(x_train[50])

plt.imshow(x_train[50])
plt.show()
print(y_train[50])

print(np.unique(y_test))
print(np.unique(y_train))

x_train=x_train/255.0
x_test=x_test/255.0

print(x_train[50])

# setting up the layers of the Neural  Network

model = keras.Sequential([
                          keras.layers.Flatten(input_shape=(28,28)),
                          keras.layers.Dense(50, activation='relu'),
                          keras.layers.Dense(50, activation='relu'),
                          keras.layers.Dense(10, activation='sigmoid')
])

# compiling the Neural Network

model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=10)

loss,accuracy=model.evaluate(x_test,y_test)
print(accuracy)

print(x_test.shape)

plt.imshow(x_test[0])

print(y_test[0])

y_pred=model.predict(x_test)

print(y_pred.shape)

print(y_pred[0])

label_for_first_test_image = np.argmax(y_pred[0])
print(label_for_first_test_image)

y_pred_labels = [np.argmax(i) for i in y_pred]
print(y_pred_labels)

conf_mat = confusion_matrix(y_test, y_pred_labels)

print(conf_mat)

plt.figure(figsize=(15,7))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')

input_image_path = '/content/MNIST_digit.png'

input_image = cv2.imread(input_image_path)

type(input_image)

print(input_image)

cv2_imshow(input_image)

input_image.shape

grayscale = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

grayscale.shape

input_image_resize = cv2.resize(grayscale, (28, 28))

input_image_resize.shape

cv2_imshow(input_image_resize)

input_image_resize = input_image_resize/255

type(input_image_resize)

image_reshaped = np.reshape(input_image_resize, [1,28,28])

input_prediction = model.predict(image_reshaped)
print(input_prediction)

input_pred_label = np.argmax(input_prediction)

print(input_pred_label)

input_image_path = input('Path of the image to be predicted: ')

input_image = cv2.imread(input_image_path)

cv2_imshow(input_image)

grayscale = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

input_image_resize = cv2.resize(grayscale, (28, 28))

input_image_resize = input_image_resize/255

image_reshaped = np.reshape(input_image_resize, [1,28,28])

input_prediction = model.predict(image_reshaped)

input_pred_label = np.argmax(input_prediction)

print('The Handwritten Digit is recognised as ', input_pred_label)