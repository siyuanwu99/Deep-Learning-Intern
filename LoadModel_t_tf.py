import tensorflow as tf
import matplotlib.pylab as plt
import numpy as np
import scipy 
from scipy import ndimage



fname = "5.jpeg"
image = np.array(ndimage.imread(fname, flatten=False))
print(np.shape(image))
my_image = np.zeros([1,64,64,3])
my_image[0,:,:,:] = scipy.misc.imresize(image, size=(64,64))/255
plt.imshow(my_image[0,:,:,:])
plt.show()



'Placeholder'

x = tf.placeholder(tf.float32, [None,64,64,3], 'X')


'Initialize filter'

W1 = tf.Variable(tf.random_normal([4,4,3,8]), name='W1', dtype=tf.float32)
W2 = tf.Variable(tf.random_normal([2,2,8,16]), name='W2', dtype=tf.float32)
W3 = tf.Variable(tf.random_normal([64,16]), name='W3', dtype=tf.float32)
W4 = tf.Variable(tf.random_normal([16,6], stddev=tf.sqrt(1/4096)), name='W4', dtype=tf.float32)

print(tf.shape(x))
'Forward propogation'
'X --- A1 --- B1 --- Z1 --- A2 --- B2 --- Z2 --- Z3 --- pred'
A1 = tf.nn.conv2d(x, W1, [1,1,1,1], padding='SAME')
B1 = tf.nn.relu(A1)
Z1 = tf.nn.max_pool(B1, [1,8,8,1], [1,8,8,1], padding='SAME')

A2 = tf.nn.conv2d(Z1, W2, [1,1,1,1], padding='SAME')
B2 = tf.nn.relu(A2)
Z2 = tf.nn.max_pool(B2, [1,4,4,1], [1,4,4,1], padding='SAME')

Z3 = tf.layers.flatten(Z2)# (n,64)
A3 = tf.matmul(Z3,W3)
Z4 = tf.nn.leaky_relu(A3, 0.1)
A4 = tf.matmul(Z4,W4)
pred = tf.nn.relu(A4)
Y_predict = tf.arg_max(tf.nn.softmax(pred),1)




with tf.Session() as sess:
    new_saver = tf.train.Saver()
    new_saver.restore(sess, 'my-test-model-170')
    print(sess.run([pred, tf.nn.softmax(pred), Y_predict],feed_dict={x: my_image}))
    


   
    

