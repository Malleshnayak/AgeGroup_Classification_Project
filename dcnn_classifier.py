    #DCNN Model for fb data
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import np_utils
from keras import backend as K
#K.set_image_dim_ordering('th')

seed = 7
numpy.random.seed(seed)


#----------Reading fb data from file-------------------

#training dataset

df=pd.read_csv("fb_data_sheet_out_final.csv")
df2=df.head(3200)
dataset=df2.values
X_train=dataset[:,0:8]
y_train=dataset[:,8]
print(len(dataset))

#Validation dataset

#df=pd.read_csv("fb_data_sheet_out_test.csv")
df3=df.tail(800)
dataset=df3.values
X_test=dataset[:,0:8]
y_test=dataset[:,8]


#Initialize number of classes i.e 2 (teenager and adult)

num_classes=2

#Reshape Training data 
X_train = X_train.reshape((3200, 8, 1))

#Reshape Testing data
X_test = X_test.reshape((800, 8, 1))


#create architecture for the model
model = Sequential()

#add convolution 1d layer
model.add(Conv1D(input_shape=X_train.shape[1:],filters=2, kernel_size=2, strides=1, padding='valid', dilation_rate=1, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

#add  dropout layer
model.add(Dropout(0.2))

#add convolutional 1d layer
model.add(Conv1D(input_shape=X_train.shape[1:],filters=2, kernel_size=2, strides=1, activation='relu', padding='same', kernel_constraint=maxnorm(3)))

#add max pooling layer for downsampling
model.add(MaxPooling1D(pool_size=2, strides=1,padding='valid'))

#add fullyconnected layer
model.add(Dense(512, activation='relu'  ))#, kernel_constraint=maxnorm(3)))

#add dropout layer
model.add(Dropout(0.5))

#add flatten layer to reduce the dimensions
model.add(Flatten())

#add fullyconnected layer
model.add(Dense(1, activation='sigmoid'))


# Compile model
print("Model...........")

# Initialize number of epochs
epochs = 50

#Initialize Learning rate
lrate = 0.01

#Initialize Decay rate
decay = lrate/epochs
#decay=0.001


sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

#Compile model
#Define Loss function and Optimizer

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Print Model summary
print(model.summary())

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=20)


# Final evaluation of the model
predicted=model.predict_classes(X_test,verbose=1)
scores = model.evaluate(X_test, y_test, verbose=0)


#Print Accracy
print("\nAccuracy: %.2f%% " % (scores[1]*100))
print("\nPrecision:Recall:f-measure")
print("Precision :")
print(precision_recall_fscore_support(y_test, predicted, average='micro'))
print(scores[0])
