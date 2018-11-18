import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3
#打印出准备训练的样本
fig = plt.figure(1)  
plt.xlabel('X')  
#设置Y轴标签  
plt.ylabel('Y')    
plt.scatter(x_data,y_data,c = 'r',marker = 'x')  #'o'---圆点，‘s’---方块
#设置图标 ,左上角的图标 
plt.legend('x')  
#显示所画的图  
plt.show()  
### tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))
y = Weights*x_data + biases
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)#梯度下降，学习速率为0.5
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
###  tensorflow structure end ###

sess = tf.Session()
sess.run(init)
for step in range(201):
	sess.run(train)
	if step % 20 == 0:
		print(step, sess.run(Weights),sess.run(biases))
