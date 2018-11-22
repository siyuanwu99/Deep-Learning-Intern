'''
This is a program from deeplearning.ai.
In this program, we calculate the convolution of pictures using the function in tensorflow.  
The following codes aim to recognize the gesture of fingers using convolutional neural networks. 
The data set is in the module named conv_utils. 
By import this, you can easily load in the data you need.



This model can be saved into disk.
 
'''

import tensorflow as t
import matplotlib.pylab as plt
import math
import numpy as np
import h5py
import scipy
from PIL import Image
from scipy import ndimage
#from tensorflow.python.framework import ops
from cnn_utils_DLCoursera import *

' import data '
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T






def cnn_modul(X_train,Y_train,X_test,Y_test,
            learning_rate = 0.001,num_epoches = 100, minibatch_size = 64):
    """
    Inplements of 3-layer ConvNet in Tensorflow
    CONV2D --> RELU --> MAX_POOL --> CONV2D --> RELU --> MAX_POOL --> FLATTEN --> FULLYCONNECTED


    Inputs:
    X_train --- input of training set, of shape (n,64,64,3)
    Y_train --- output of training set, of shape (n,6)
    X_test  --- input of test set, of shape (n,64,64,3)
    Y_test  --- output of test set, of shape (n,6)
    
    Hyper parameters:
    learning_rate   --- learning rate of optimization
    num_epoches     --- numbers of epoches of optimization loop
    minibatch_size  --- size of minibatch
    
    Returns:
    train accuracy  --- scalar
    test accuracy   --- scalar
    parameters      --- python dictionary

    """

    (m, n_H, n_W, n_C) = np.shape(X_train)
    (m,n_y) = np.shape(Y_train)
    costs_curv = []

    
    tf.set_random_seed(1)

    'Placeholder'
    X = tf.placeholder(tf.float32, [None,n_H,n_W,n_C], 'X')
    Y = tf.placeholder(tf.float32, [None,6], 'Y')


    'Initialize filter'
    W1 = tf.Variable(tf.random_normal([4,4,3,8]), name='W1', dtype=tf.float32)
    W2 = tf.Variable(tf.random_normal([2,2,8,16]), name='W2', dtype=tf.float32)
    W3 = tf.Variable(tf.random_normal([64,16]), name='W3', dtype=tf.float32)
    W4 = tf.Variable(tf.random_normal([16,6], stddev=tf.sqrt(1/4096)), name='W4', dtype=tf.float32)
    #b3 = tf.Variable(tf.random_normal([1,16]),name='b3',dtype=tf.float32)
    #b4 = tf.Variable(tf.random_normal([1,6]),name='b4',dtype=tf.float32)


    
    parameters = {  'W1': W1,
                    'W2': W2,
                    'W3': W3,
                    'W4': W4    }


    'Forward propogation'
    'X --- A1 --- B1 --- Z1 --- A2 --- B2 --- Z2 --- Z3 --- pred'
    A1 = tf.nn.conv2d(X, W1, [1,1,1,1], padding='SAME')
    B1 = tf.nn.relu(A1)
    Z1 = tf.nn.max_pool(B1, [1,8,8,1], [1,8,8,1], padding='SAME')
    
    A2 = tf.nn.conv2d(Z1, W2, [1,1,1,1], padding='SAME')
    B2 = tf.nn.relu(A2)
    Z2 = tf.nn.max_pool(B2, [1,4,4,1], [1,4,4,1], padding='SAME')

    Z2 = tf.layers.flatten(Z2)
    A3 = tf.matmul(Z2,W3)
    Z4 = tf.nn.leaky_relu(A3, 0.1)
    A4 = tf.matmul(Z4,W4)
    pred = tf.nn.relu(A4)
   
   

    'Cost'
    cost_sum = tf.nn.softmax_cross_entropy_with_logits(labels= Y, logits= pred)
    cost = tf.reduce_sum(cost_sum)


    'Optimizer'
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


    'initializer'
    init = tf.global_variables_initializer()


    'saver class'
    saver = tf.train.Saver()


    'Allon - sy'
    with tf.Session() as sess :
                
        seed = 1
        sess.run(init)

        saver.save(sess,"./my-test-model",global_step=0)
        
        #training loop
        for epoch in range(num_epoches):
            minibatch_cost = 0
            num_minibatch = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, cost_temp = sess.run([optimizer, cost],feed_dict= {X: minibatch_X, Y: minibatch_Y})
                
                minibatch_cost += cost_temp / minibatch_size

            if epoch % 10 == 0 and epoch > 4:
                print("Cost after epoch %i : %f"%(epoch,minibatch_cost))
            elif epoch % 1 == 0:
                costs_curv.append(minibatch_cost)

        'Save the model'
        saver.save(sess,'./my-test-model',global_step=170)
        print("Successfully saved the model!!")
      
        'Calculate the correct predictions'
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
            

        'Calculate accuracy on the test set'
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)


        plt.plot(np.squeeze(costs_curv))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()         
        
        par = sess.run(parameters)
        print(par)
        return par




par = cnn_modul(X_train,Y_train,X_test,Y_test,0.01,170,64)

