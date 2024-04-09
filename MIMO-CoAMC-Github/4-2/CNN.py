# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 16:38:26 2019

@author: Rain
"""
from keras import models
from keras.layers import Flatten, Dense, BatchNormalization, Dropout, Conv1D
Rt = 2 #transmitter
Rr = 4 #receicer
N = int(128/Rt) #sample points
model = models.Sequential()
model.add(Conv1D(128, 16, activation='relu', padding='same',input_shape=[N, 2]))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Conv1D(64, 8, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam', 
              metrics=['accuracy'])
model.summary()


import scipy.io as scio
import numpy as np
from numpy import array
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix
snrs=range(-10, 12, 2)
for snr in snrs:
  data_path="Dataset/IQ/Original/train/"+str(snr)+".mat"
  data = scio.loadmat(data_path)
  x1 = data.get('IQ')
  x = x1[:,:,np.newaxis] 
  x_real = x1.real
  x_imag = x1.imag
  x =  np.concatenate((x_real, x_imag), axis = 2)
  train_num_per_modulation =20000
  y1=np.zeros([train_num_per_modulation*Rr,1])
  y2=np.ones([train_num_per_modulation*Rr,1])
  y3=np.ones([train_num_per_modulation*Rr,1])*2
  y4=np.ones([train_num_per_modulation*Rr,1])*3
  y=np.vstack((y1,y2,y3,y4))
  y = array(y)
  y = to_categorical(y)
  X_train, X_val, Y_train, Y_val = train_test_split(x, 
                                                    y, 
                                                    test_size = 0.3, 
                                                    random_state= 30)
  
  checkpoint = ModelCheckpoint("Model/CNN/"+str(snr)+".hdf5", 
                            verbose=1, 
                            save_best_only=True)
  tensorboard = TensorBoard("Model/CNN/"+str(snr)+".log", 0)
  earlystopping = EarlyStopping(monitor="val_loss", 
	                              patience=10, 
	                              verbose=1, 
	                              mode="auto")

  model.fit(X_train,
            Y_train,
            batch_size=500,
            epochs=100,
            verbose=1,
            validation_data=(X_val, Y_val),
            callbacks=[checkpoint, tensorboard, earlystopping]
            )
#################test#####################
for snr in snrs:
   model.load_weights("Model/CNN/"+str(snr)+".hdf5")
   data_path="Dataset/IQ/Original/test/"+str(snr)+".mat"
   data = scio.loadmat(data_path)
   x = data.get('IQ')
   x = x[:,:,np.newaxis] 
   x_real = x.real
   x_imag = x.imag
   x =  np.concatenate((x_real, x_imag), axis = 2)
   test_num_per_modulation = 10000
   y1 = np.zeros([test_num_per_modulation*Rr,1])
   y2 = np.ones([test_num_per_modulation*Rr,1])
   y3 = np.ones([test_num_per_modulation*Rr,1])*2
   y4 = np.ones([test_num_per_modulation*Rr,1])*3
   y = np.vstack((y1,y2,y3,y4))
   y = array(y)
   y = to_categorical(y)
   X_test=x
   Y_test=y
   [loss, acc] = model.evaluate(X_test,Y_test, batch_size=1000, verbose=0)
   print(acc)

#################average weighting#####################
for snr in snrs:
   model.load_weights("Model/CNN/"+str(snr)+".hdf5")
   data_path="Dataset/IQ/Original/test/"+str(snr)+".mat"
   data = scio.loadmat(data_path)
   x = data.get('IQ')
   x = x[:,:,np.newaxis] 
   x_real = x.real
   x_imag = x.imag
   x =  np.concatenate((x_real, x_imag), axis = 2)
   test_num_per_modulation = 10000
   X_test=x
   proba = model.predict_proba(X_test)
   temp = np.zeros([test_num_per_modulation*Rr,4])
   for i in range(test_num_per_modulation*Rr):
       temp[i,:] = proba[4*i,:] + proba[4*i+1,:] + proba[4*i+2,:] + proba[4*i+3,:]
   proba_ensemble = temp
   predict_lables = []
   for j in proba_ensemble:
       tmp = np.argmax(j, 0)
       predict_lables.append(tmp)

   y1 = np.zeros([test_num_per_modulation,1])
   y2 = np.ones([test_num_per_modulation,1])
   y3 = np.ones([test_num_per_modulation,1])*2
   y4 = np.ones([test_num_per_modulation,1])*3
   y = np.vstack((y1,y2,y3,y4))
   true_lables = array(y)
   cm = confusion_matrix(true_lables, predict_lables)
   acc_ensemble = (cm[0,0] + cm[1,1] + cm[2,2] + cm[3,3])/(test_num_per_modulation*4)
   print(acc_ensemble)

################voting#####################
for snr in snrs:
   model.load_weights("Model/CNN/"+str(snr)+".hdf5")
   data_path="../Dataset/IQ/Original/test/"+str(snr)+".mat"
   data = scio.loadmat(data_path)
   x = data.get('IQ')
   x = x[:,:,np.newaxis] 
   x_real = x.real
   x_imag = x.imag
   x =  np.concatenate((x_real, x_imag), axis = 2)
   test_num_per_modulation = 10000
   X_test=x
   classes = model.predict_classes(X_test)
   classes = to_categorical(classes)
   temp = np.zeros([test_num_per_modulation*Rr,4])
   for i in range(test_num_per_modulation*Rr):
       temp[i,:] = classes[4*i,:] + classes[4*i+1,:] + classes[4*i+2,:] + classes[4*i+3,:]
   classes_ensemble = temp
   predict_lables = []
   for j in classes_ensemble:
       tmp = np.argmax(j, 0)
       predict_lables.append(tmp)

   y1 = np.zeros([test_num_per_modulation,1])
   y2 = np.ones([test_num_per_modulation,1])
   y3 = np.ones([test_num_per_modulation,1])*2
   y4 = np.ones([test_num_per_modulation,1])*3
   y = np.vstack((y1,y2,y3,y4))
   true_lables = array(y)
   cm = confusion_matrix(true_lables, predict_lables)
   acc_ensemble = (cm[0,0] + cm[1,1] + cm[2,2] + cm[3,3])/(test_num_per_modulation*4)
   print(acc_ensemble)
#######################weight average###################################################################
for snr in snrs:
	data_path="Dataset/IQ/Original/train/"+str(snr)+".mat"
	data = scio.loadmat(data_path)
	x = data.get('IQ')
	x = x[:,:,np.newaxis] 
	x_real = x.real
	x_imag = x.imag
	x =  np.concatenate((x_real, x_imag), axis = 2)
	train_num_per_modulation =20000
	y1=np.zeros([train_num_per_modulation*Rr,1])
	y2=np.ones([train_num_per_modulation*Rr,1])
	y3=np.ones([train_num_per_modulation*Rr,1])*2
	y4=np.ones([train_num_per_modulation*Rr,1])*3
	y=np.vstack((y1,y2,y3,y4))
	y = array(y)
	y = to_categorical(y)
	x1 = []
	x2 = []
	x3 = []
	x4 = []
	y1 = []
	y2 = []
	y3 = []
	y4 = []
	for i in range(train_num_per_modulation*Rr):
		x1.append(x[4*i,:,:])
		x2.append(x[4*i+1,:,:])
		x3.append(x[4*i+2,:,:])
		x4.append(x[4*i+3,:,:])
		y1.append(y[4*i,:])
		y2.append(y[4*i+1,:])
		y3.append(y[4*i+2,:])
		y4.append(y[4*i+3,:])
	x1 = array(x1)
	x2 = array(x2)
	x3 = array(x3)
	x4 = array(x4)
	y1 = array(y1)
	y2 = array(y2)
	y3 = array(y3)
	y4 = array(y4)
	model.load_weights("Model/CNN/"+str(snr)+".hdf5")
	[loss1,acc1] = model.evaluate(x1,y1, batch_size=1000, verbose=0)
	[loss2,acc2] = model.evaluate(x2,y2, batch_size=1000, verbose=0)
	[loss3,acc3] = model.evaluate(x3,y3, batch_size=1000, verbose=0)
	[loss4,acc4] = model.evaluate(x4,y4, batch_size=1000, verbose=0)
	ratio1 = acc1/(acc1+acc2+acc3+acc4)
	ratio2 = acc2/(acc1+acc2+acc3+acc4)
	ratio3 = acc3/(acc1+acc2+acc3+acc4)
	ratio4 = acc4/(acc1+acc2+acc3+acc4)
	data_path="Dataset/IQ/Original/test/"+str(snr)+".mat"
	data = scio.loadmat(data_path)
	x = data.get('IQ')
	x = x[:,:,np.newaxis] 
	x_real = x.real
	x_imag = x.imag
	x =  np.concatenate((x_real, x_imag), axis = 2)
	test_num_per_modulation = 10000
	X_test=x
	proba = model.predict_proba(X_test)
	temp = np.zeros([test_num_per_modulation*Rr,4])
	for i in range(test_num_per_modulation*Rr):
	    temp[i,:] = ratio1*proba[4*i,:] + ratio2*proba[4*i+1,:] + ratio3*proba[4*i+2,:] + ratio4*proba[4*i+3,:]
	proba_ensemble = temp
	predict_lables = []
	for j in proba_ensemble:
	    tmp = np.argmax(j, 0)
	    predict_lables.append(tmp)

	y1 = np.zeros([test_num_per_modulation,1])
	y2 = np.ones([test_num_per_modulation,1])
	y3 = np.ones([test_num_per_modulation,1])*2
	y4 = np.ones([test_num_per_modulation,1])*3
	y = np.vstack((y1,y2,y3,y4))
	true_lables = array(y)
	cm = confusion_matrix(true_lables, predict_lables)
	acc_ensemble = (cm[0,0] + cm[1,1] + cm[2,2] + cm[3,3])/(test_num_per_modulation*4)
	print(acc_ensemble)
#######################weight vote###################################################################
for snr in snrs:
 	data_path="Dataset/IQ/Original/train/"+str(snr)+".mat"
 	data = scio.loadmat(data_path)
 	x = data.get('IQ')
 	x = x[:,:,np.newaxis] 
 	x_real = x.real
 	x_imag = x.imag
 	x =  np.concatenate((x_real, x_imag), axis = 2)
 	train_num_per_modulation =20000
 	y1=np.zeros([train_num_per_modulation*Rr,1])
 	y2=np.ones([train_num_per_modulation*Rr,1])
 	y3=np.ones([train_num_per_modulation*Rr,1])*2
 	y4=np.ones([train_num_per_modulation*Rr,1])*3
 	y=np.vstack((y1,y2,y3,y4))
 	y = array(y)
 	y = to_categorical(y)
 	x1 = []
 	x2 = []
 	x3 = []
 	x4 = []
 	y1 = []
 	y2 = []
 	y3 = []
 	y4 = []
 	for i in range(train_num_per_modulation*Rr):
 		x1.append(x[4*i,:,:])
 		x2.append(x[4*i+1,:,:])
 		x3.append(x[4*i+2,:,:])
 		x4.append(x[4*i+3,:,:])
 		y1.append(y[4*i,:])
 		y2.append(y[4*i+1,:])
 		y3.append(y[4*i+2,:])
 		y4.append(y[4*i+3,:])
 	x1 = array(x1)
 	x2 = array(x2)
 	x3 = array(x3)
 	x4 = array(x4)
 	y1 = array(y1)
 	y2 = array(y2)
 	y3 = array(y3)
 	y4 = array(y4)
 	model.load_weights("Model/CNN/"+str(snr)+".hdf5")
 	[loss1,acc1] = model.evaluate(x1,y1, batch_size=1000, verbose=0)
 	[loss2,acc2] = model.evaluate(x2,y2, batch_size=1000, verbose=0)
 	[loss3,acc3] = model.evaluate(x3,y3, batch_size=1000, verbose=0)
 	[loss4,acc4] = model.evaluate(x4,y4, batch_size=1000, verbose=0)
 	ratio1 = acc1/(acc1+acc2+acc3+acc4)
 	ratio2 = acc2/(acc1+acc2+acc3+acc4)
 	ratio3 = acc3/(acc1+acc2+acc3+acc4)
 	ratio4 = acc4/(acc1+acc2+acc3+acc4)
 	data_path="Dataset/IQ/Original/test/"+str(snr)+".mat"
 	data = scio.loadmat(data_path)
 	x = data.get('IQ')
 	x = x[:,:,np.newaxis] 
 	x_real = x.real
 	x_imag = x.imag
 	x =  np.concatenate((x_real, x_imag), axis = 2)
 	test_num_per_modulation = 10000
 	X_test=x
 	classes = model.predict_classes(X_test)
 	classes = to_categorical(classes)
 	temp = np.zeros([test_num_per_modulation*Rr,4])
 	for i in range(test_num_per_modulation*Rr):
 	   temp[i,:] = ratio1*classes[4*i,:] + ratio2*classes[4*i+1,:] + ratio3*classes[4*i+2,:] + ratio4*classes[4*i+3,:]
 	classes_ensemble = temp
 	predict_lables = []
 	for j in classes_ensemble:
 	   tmp = np.argmax(j, 0)
 	   predict_lables.append(tmp)
 	y1 = np.zeros([test_num_per_modulation,1])
 	y2 = np.ones([test_num_per_modulation,1])
 	y3 = np.ones([test_num_per_modulation,1])*2
 	y4 = np.ones([test_num_per_modulation,1])*3
 	y = np.vstack((y1,y2,y3,y4))
 	true_lables = array(y)
 	cm = confusion_matrix(true_lables, predict_lables)
 	acc_ensemble = (cm[0,0] + cm[1,1] + cm[2,2] + cm[3,3])/(test_num_per_modulation*4)
 	print(acc_ensemble)
    
