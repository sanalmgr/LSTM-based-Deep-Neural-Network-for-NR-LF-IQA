"""
@author: sanaalamgeer
"""
import numpy as np
import cv2
from keras.applications.vgg16 import VGG16, preprocess_input

left = np.load('variables/stream2_train_lfdd_mli.npz') #train
#left = np.load('Bottleneck/stream1_test_mpi_mli_256x256.npz') #test
X_trainLeft = left['image_left']

width, height = 256, 256 

model = VGG16(weights='imagenet', include_top=False, input_shape=(height, width, 3))

#Preprocessing the data, so that it can be fed to the pre-trained vgg16 model. 
resnet_train_input = preprocess_input(X_trainLeft)
print(resnet_train_input.shape)

#Creating bottleneck features for the training data
train_features = model.predict(resnet_train_input)
print(train_features.shape)

train_features = train_features.reshape(train_features.shape[0], train_features.shape[-1], train_features.shape[1], train_features.shape[2]) #vgg16

print(train_features.shape)

#Saving the bottleneck features
filename = 'Bottleneck/mli_image_train'
np.savez(filename, features = train_features)
print('Done!')
