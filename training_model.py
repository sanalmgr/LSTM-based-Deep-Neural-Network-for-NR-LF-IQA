"""
@author: sanaalamgeer
"""
import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Conv2D, ELU, MaxPooling2D, Flatten, Dense, Dropout, LSTM, Reshape
from keras.optimizers import SGD

#%%
def get_model(width, height):
	#Stream 1
	stream1_image = Input(shape=(width, height, 3))
	#conv1
	stream1_conv1 = Conv2D(32, (3, 3), padding='same', name='conv1_stream1')(stream1_image)
	stream1_elu1 = ELU()(stream1_conv1)
	stream1_pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1_stream1')(stream1_elu1)
	#Reshape
	stream1_reshape = Reshape((-1, stream1_pool1.shape[2]))(stream1_pool1)
	#lstm1
	stream1_lstm1 = LSTM(units = 5, return_sequences = True, name='stream1_lstm1')(stream1_reshape)
	stream1_drop1 = Dropout(0.5)(stream1_lstm1)
	#fc6
	stream1_flat6 = Flatten()(stream1_drop1)
	stream1_fc6 = Dense(512)(stream1_flat6) #512
	stream1_elu6 = ELU()(stream1_fc6)
	stream1_drop6 = Dropout(0.5)(stream1_elu6)
	#fc7
	stream1_fc7 = Dense(512)(stream1_drop6) #512
	stream1_elu7 = ELU()(stream1_fc7)
	stream1_drop7 = Dropout(0.5)(stream1_elu7)
	
	#Second Input
	aux_input = Input(shape=(512, 8, 8)) # vgg16 
	#Reshape
	aux_input_flat = Flatten()(aux_input)
	aux_input_fc6 = Dense(512)(aux_input_flat) #512
	aux_input_elu6 = ELU()(aux_input_fc6)
	aux_input_drop6 = Dropout(0.5)(aux_input_elu6)
	#Adding two inputs
	add_conv2 = keras.layers.add([stream1_drop7, aux_input_drop6])
	fusion1_add_elu2 = ELU()(add_conv2)
	#fc9
	fusion3_fc8 = Dense(1024)(fusion1_add_elu2)
	#fc9
	predictions = Dense(1)(fusion3_fc8)
	
	model_all = Model([stream1_image, aux_input], predictions, name = 'all_model')
	
	return model_all

def compile_model(model):
	sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=1)
	model.compile(loss='mean_squared_error', optimizer=sgd)
	return model

def fit_model(model, X_train, bottleneck_features, Y_train, model_checkpoint, bs=128, eps=6000):
	history = model.fit(x = [X_train, bottleneck_features], y = [Y_train], validation_split=0.2, batch_size=bs, epochs=eps, verbose=1, callbacks=[model_checkpoint], shuffle=True)
	
	#saving history
	np.save('train_history.npy',history.history)
	
	print('complete training!')

#END
