import numpy as np 
import c_csvread as c
import matplotlib.pyplot as plt 

x_database,y_database = c.my_read("winequality-red.csv")



y_database = (y_database > np.mean(y_database))
x_data = x_database[0:1000,[3,7]]
y_data = y_database[0:1000,:]
x_test = x_database[1000:1200,[3,7]]
y_test = y_database[1000:1200,:]

x_data_stand = (x_data-np.mean(x_data))/np.std(x_data)



#Let's create a neutral network


def my_sigmoid(z):
    y = 1/(1+np.exp(-z))
    return y

def my_tanh(z):
    y = (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
    return y

def my_ReLU(z):
    y = np.maximum(0,z)
    return z



def d_ReLU(z):
    result = (z >= 0)
    return result

def d_sigmoid(z):
    result = my_sigmoid(z)*(1-my_sigmoid(z))
    return result

def d_tanh(z):
    result = 1-my_tanh(z)**2
    return result

def loss_func(input,Y):
    m = np.size(Y,axis=0)
    loss = sum((input-Y)**2)/m/2
    return loss

def d_loss_func(pred,Y):
    return pred-Y

def cost_func(inp,Y):
    cost =0.5 * np.linalg.norm(inp - Y)
    return cost

    

def backforward(x_1,y_1,w,v,b1,b2): #反向传播，返回更新后的参数
    
    global learning_rate
   
    def grediant_v(p,z2,y):
        result = (p-y)*d_sigmoid(z2)
        return result

    def grediant_w(g_layer,v,z1):
        result = g_layer*(np.transpose(d_tanh(z1))*v)
        return result
    
    z1 = x_1.dot(w) + b1 #n-by-3
    a1 = my_tanh(z1)    #n-by-3
    z2 = a1.dot(v) + b2 #n-by-1 v:3-by-1
    pred = my_sigmoid(z2)   #n-by-1
    cost = cost_func(pred,y_1)

    g_v_layer = grediant_v(pred,z2,y_1) #n-by-1
    g_v = np.dot(a1.transpose(),g_v_layer) #g_v_layer*np.transpose(a1) 3-by-1
    g_b2 = np.sum(g_v_layer)

    g_w_layer = grediant_w(g_v_layer,v,z1)  #2-by-3
    g_w = g_w_layer.dot(x_1)    
    g_b1  = sum(g_w_layer)

    w = w - learning_rate*np.transpose(g_w)
    v = v - learning_rate*g_v
    b1 = b1 - g_b1
    b2 = b2 - g_b2

    return w ,v,b1,b2, cost


def forward(w,v,b1,b2,x_1): #前向传播2-3-1型网络
    z1 = x_1.dot(w) + b1
    a1 = my_tanh(z1)
    z2 = a1.dot(v) + b2
    pred = my_sigmoid(z2)
    if pred>0.5:
        pred = 1
    else:
        pred = 0
    return pred

# Y = np.array([[1],[3],[4]])
# pred = np.array([[2],[2],[3]])
# print(loss_func(pred,Y ))



# init

w_init = np.random.randn(2,3)
v_init = np.random.randn(3,1)

print(w_init,v_init)
learning_rate = 0.01

# x1 = x_data_stand[1,:][np.newaxis,:]
# y1 = y_database[1]
# w,v,b1,b2,cost = backforward(x1,y1,w_init,v_init,1,1)
# cost = np.zeros([500,1])


w = w_init
v = v_init
b1 = 0
b2 = 0


for index in range(1,500):

   # wi,vi,b1i,b2i,cost[index] = backforward(x_data_stand[index,:][np.newaxis,:],y_database[index],w,v,b1,b2)
    wi, vi, b1i, b2i, cost = backforward(x_data_stand,y_data,w,v,b1,b2)
    #print('w1',wi)
    #print("v1",vi)
    #print("cost",cost[index])
    w = wi
    v = vi
    b1 = b1i
    b2 = b2i

#模型测试
pred = np.ones([100,1])
test = np.zeros([100,1])
a = 550
# for index in range(a,a+100):
#     pred[index-a] = forward(w,v,b1,b2,x_data_stand[index,:][np.newaxis,:])
  
#     if pred[index-a] == y_database[index]:
#         test[index-a] = 1

pred  =  forward(w,v,b1,b2,x_test)

pred1 = (pred > 0.5)

correct = (pred1 == y_test)

accuracy = np.sum(correct)/np.size(correct)

print(accuracy)

    







plt.figure(1)
plt.plot(cost)
plt.show()



# print('w1',w1)
# print("v1",v1)
# print("cost",cost)