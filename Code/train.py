"""
@author: sanaalamgeer
"""
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input,Conv2D,ELU,Softmax,MaxPooling2D,Flatten,Dense,Dropout,normalization, LSTM, Reshape
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.optimizers import SGD
from keras import backend as K
from keras.callbacks import CSVLogger
import os

height, width  = 81, 654 #960, 100

#%%###
#%% Load dataset
left = np.load('variables/stream1_horizontal_epi_train.npz')

X_trainLeft = left['image_left']
Y_trainLeft = left['label_left']

X_trainLeft = X_trainLeft.astype('float32')
X_trainLeft /= 255

#extracting bottleneck features
btlneck = np.load('Bottleneck/mli_image_train.npz')
train_features = btlneck['features']

#%%
#######################################################################################
#left image
left_image=Input(shape=(width, height, 3))
#conv1
left_conv1=Conv2D(32, (3, 3), padding='same', name='conv1_left')(left_image)
left_elu1=ELU()(left_conv1)
left_pool1=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1_left')(left_elu1)
#Reshape
left_reshape = Reshape((-1, left_pool1.shape[2]))(left_pool1)
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

model_all = Model([left_image, aux_input], predictions, name = 'all_model')

model_all.summary()
#######################################################################################
#%% train model
sgd=SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=1) #lr=0.0001
model_all.compile(loss='mean_squared_error', optimizer=sgd, metrics=["accuracy"])
#######################################################################################
# simple early stopping
#es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=200)
mc = ModelCheckpoint('model/vgg16_lfdd.hdf5', monitor='loss', mode='min', verbose=1, save_best_only=True)
#lrschd = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)
#reduce_lr = ReduceLROnPlateau(monitor='loss', factor=tf.math.exp(-0.1), patience=100, min_lr=0.0001, verbose=1)
#######################################################################################
#fitting model
csv_logger = CSVLogger('vgg16_lfdd.csv', append=True, separator=';')
history = model_all.fit(x = [X_trainLeft, train_features], y = [Y_trainLeft], validation_split=0.2, batch_size=128, epochs=6000, verbose=1, callbacks=[mc, csv_logger], shuffle=True)
#######################################################################################
#saving history
np.save('train_history.npy',history.history)

#model_all.save_weights('model/trained_model.hdf5')
print('complete training!')
#END
#######################################################################################
