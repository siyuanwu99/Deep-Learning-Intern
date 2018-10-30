import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
## Using tf to learn a linear regression model 

## initial data
x_data = np.random.rand(100)
y_data = 0.1*x_data + 0.3 + 0.03*np.random.rand(100)

#plt.figure(1)
#plt.plot(x_data,y_data,'ro')
#plt.show()

## tf

W = tf.Variable([np.random.uniform(0,2)])
b = tf.Variable(tf.zeros([1]))
pred = W*x_data + b
loss = tf.reduce_mean(tf.square(pred - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

## Let's start trainning
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print('loss=',sess.run(loss),'Weight= ',sess.run(W),'biases=',sess.run(b))
for index in range(1,40):
    sess.run(train)
    print('loss=',sess.run(loss),'Weight= ',sess.run(W),'biases=',sess.run(b))
plt.figure(index)
x1 = np.arange(0.,1.,0.1)
y1 = sess.run(W)*x1 + sess.run(b)
plt.plot(x1,y1,'b-')
plt.plot(x_data,y_data,'ro')
plt.show()
plt.close()

sess.close()



