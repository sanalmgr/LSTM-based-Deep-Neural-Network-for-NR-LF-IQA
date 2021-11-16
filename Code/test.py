"""
@author: sanaalamgeer
"""
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input,Conv2D,ELU,MaxPooling2D,Flatten,Dense,Dropout,normalization, LSTM, Reshape
from keras.optimizers import SGD
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
import os
import scipy.io as sio
from keras.callbacks import CSVLogger

height, width  = 81, 512 #960, 100

#%%###
#%% Load dataset
left = np.load('variables/stream1_horizontal_epi_test.npz')

X_testLeft = left['image_left']
Y_testLeft = left['label_left']

X_testLeft = X_testLeft.astype('float32')
X_testLeft /= 255

#extracting bottleneck features
btlneck = np.load('Bottleneck/mli_image_test.npz')
test_features = btlneck['features']

#%%
#######################################################################################
#left image
left_image=Input(shape=(width, height, 3))
#conv1
left_conv1=Conv2D(32, (3, 3), padding='same', name='conv1_left')(left_image)
left_elu1=ELU()(left_conv1)
left_pool1=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1_left')(left_elu1)
#Reshape
left_reshape = Reshape((-1, width))(left_pool1)
#lstm1
left_lstm1=LSTM(units = 5, return_sequences = True, name='left_lstm1')(left_reshape)
left_drop1=Dropout(0.5)(left_lstm1)
#fc6
left_flat6=Flatten()(left_drop1)
left_fc6=Dense(512)(left_flat6) #512
left_elu6=ELU()(left_fc6)
left_drop6=Dropout(0.5)(left_elu6)
#fc7
left_fc7=Dense(512)(left_drop6) #512
left_elu7=ELU()(left_fc7)
left_drop7=Dropout(0.5)(left_elu7)

#Second Input
aux_input = Input(shape=(512, 8, 8)) # vgg16
#Reshape
aux_input_flat = Flatten()(aux_input)
aux_input_fc6 = Dense(512)(aux_input_flat) #512
aux_input_elu6 = ELU()(aux_input_fc6)
aux_input_drop6 = Dropout(0.5)(aux_input_elu6)
#Adding two inputs
add_conv2 = keras.layers.add([left_drop7, aux_input_drop6])
fusion1_add_elu2 = ELU()(add_conv2)
#fc9
fusion3_fc8 = Dense(1024)(fusion1_add_elu2)
#fc9
predictions = Dense(1)(fusion3_fc8)

model_all = Model(input = [left_image, aux_input], output = predictions, name = 'all_model')

model_all.summary()
#######################################################################################
model_all.load_weights('model/trained_model.hdf5')
y_predict = model_all.predict([X_testLeft, test_features], batch_size=10)
sio.savemat('Outputs/predictScore_vgg16.mat', {'score': y_predict})
print('complete testing!!')
##############################################################################################
##############################################################################################
##############################################################################################
