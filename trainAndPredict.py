from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import pickle
import os
import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets, svm, metrics
import pandas as pd
from keras.models import load_model
pd.options.display.float_format = '{:,.2f}'.format

cnt = 0
acc = 0
x_train = []
y_train = []
label = 0
files = []
path = './train_with_edges/'
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.jpg' in file:
                files.append(file)
                # print(file)
                gray = cv2.imread(path + file)
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(255-gray, (100, 100))
                x_train.append(gray)
                y_train.append(ord(file[0])-65)

# for code in range(ord('A'), ord('Z') + 1):
#   for j in range(1, 8):
#     exists = os.path.isfile('train_with_edges/'+chr(code)+str(j)+'.jpg')
#     if exists:
#       cnt = cnt+1
#       gray = cv2.imread('RotatedData/'+chr(code)+str(j)+'.jpg')
#       gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
#       gray = cv2.resize(255-gray, (100, 100))
#       x_train.append(gray)
#       y_train.append(label)
#   label=label+1

x_train = np.array(x_train)
y_train = np.array(y_train)
print(x_train.shape)
print(y_train.shape)
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, test_size=0.3, random_state=42)

# # print(x_train[0])
# # plt.imshow(x_train[0])
# # plt.show(1)
# # print(y_train[0])

x_train = x_train.reshape(x_train.shape[0], 100, 100, 1)
#x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (100, 100, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
#x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
#print('Number of images in x_test', x_test.shape[0])


# Importing the required Keras modules containing model and layers
# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(2, kernel_size=(3, 3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
model.add(Dense(32, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(26, activation='softmax'))


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=x_train, y=y_train, epochs=10)

model.save("model.h5")
# cnt = 0
# acc = 0
# for i in range(0, 10):
#     for j in range(1, 10):
#         exists = os.path.isfile('Dataset/i'+str(i)+str(j)+'.jpg')
#         if exists:
#             cnt = cnt+1
#             gray = cv2.imread('Dataset/i'+str(i)+str(j)+'.jpg', 1)
#             gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
#             gray = cv2.resize(255-gray, (28, 28))
#             # gray,thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
#             flatten = gray.flatten() / 255.0
#             # print(flatten)
#             plt.imshow(gray, cmap='Greys')
#             plt.show(10)
# #
#             pred = model.predict(flatten.reshape(1, 28, 28, 1))
#             print(pred.argmax())
#             if pred.argmax() == i:
#                 acc = acc+1

# gray = cv2.imread('4.jpg', 1)
# gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
# gray = cv2.resize(255-gray, (50, 50))
# # gray,thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
# flatten = gray.flatten() / 255.0
# print(flatten)
# # plt.imshow(gray, cmap='Greys')
# # plt.show(10)
# #
# # load weights into new model
# loaded_model=load_model("model.h5")
# pred = loaded_model.predict(flatten.reshape(1, 50, 50, 1))
# print(pred.argmax())

# print(acc/cnt)

# gray = cv2.imread('3.jpg', 1)
# gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
# gray = cv2.resize(255-gray, (28, 28))
# # gray,thresh = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)
# flatten = gray.flatten() / 255.0
# pred = model.predict(flatten.reshape(1, 28, 28, 1))
# print(pred.argmax())
# cnt = 0
# acc = 0
# for i in range(len(x_test)):
#     cnt = cnt+1
#     pred = model.predict(flatten.reshape(1, 28, 28, 1))
#     flatten = x_test[i].reshape(1, 28, 28, 1)
#     flatten = x_test[i].astype('float32')
#     flatten /= 255
#     pred = model.predict(flatten.reshape(1, 28, 28, 1))
#     if pred.argmax() == y_test[i]:
#         acc = acc+1
#     # print(pred.argmax())

# print(acc/cnt)
