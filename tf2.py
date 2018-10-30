import tensorflow as tf 
import numpy as np 
import csv

##
x_data=[]
y_data=[]            #定义空列表
csvFile = open(r"winequality-red.csv", "r")
reader = csv.reader(csvFile)
for item in reader:
    # if reader.line_num==1:
        # continue
    item=[float(ii) for ii in item]
    x_data.append(item)

###把读取的数据转化成float格式

for i in range(len(x_data)):
    y_data.append(x_data[i].pop())

# # print('x_data',x_data)
# # print('y_data',y_data)



x2 = np.array(x_data)
y1 = np.array(y_data)
x1 = x2[:,3:5]
# print(x1)
batch_size = 2

w1 = tf.Variable(tf.random_normal([2,3]))
w2 = tf.Variable(tf.random_normal([3,1]))

#x = tf.placeholder(tf.float32,shape=(None,2),name="imput")


a = tf.matmul(x1,w1)
a1 = tf.nn.sigmoid(a)
a2 = tf.matmul(a1,w2)
pred = tf.nn.sigmoid(a2)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
#print (sess.run(y,feed_dict={x:[[0.7,0.9],[1.0,1.5],[2.1,2.3]]}))

print(sess.run(w1))
print(sess.run(w2))

loss = -tf.reduce_mean(y1*tf.log(pred)+(1-y1)*tf.log(1-pred))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)    

STEP = 10

for i in range(0,STEP):
    sess.run(train)
    print(loss)



