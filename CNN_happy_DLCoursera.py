import numpy as np
import tensorflow as tf
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils_DLcoursera import *


import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow





X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

def HappyModel(input_shape):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    
    ### START CODE HERE ###
    # Feel free to use the suggested outline in the text above to get started, and run through the whole
    # exercise (including the later portions of this notebook) once. The come back also try out other
    # network architectures as well. 
    
    # Input Data
    X_input = Input(shape= input_shape)
    
    X = ZeroPadding2D((4,4))(X_input)       #72-72-3
    X = Conv2D(16, (8,8), strides = (1,1))(X)   #64-64-16
    X = BatchNormalization(axis= 3 )(X)
    X = Activation('relu') (X)
    X = MaxPooling2D((4,4))(X) #32-32-16

    X = ZeroPadding2D((4,4))(X)
    X = Conv2D(32, (8,8), strides= (1,1))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X =MaxPooling2D((4,4))(X)

    X = Flatten( ) (X)
    X = Dense(64, activation='relu')(X)
    X = Dense(16,activation='relu')(X)
    X = Dense(1, activation= 'sigmoid')(X)

    model = Model(inputs= X_input, outputs= X, name='HappyModel')

    return model




model = HappyModel((64, 64, 3))

import keras
model.compile(optimizer=keras.optimizers.Adam(), loss= 'binary_crossentropy', metrics=['accuracy'])

model.fit(x=X_train, y=Y_train, batch_size=64, epochs= 10)

preds = model.evaluate( x=X_test, y=Y_test)

print()
print("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
