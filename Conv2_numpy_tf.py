#
#对图像进行卷积运算
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt



def my_conv(A,B):
    '''
    convolution
    Argument:
    A -- Matrix
    B -- filter
    
    Return:
    output -- matrix

    
    '''
    A = np.array(A)
    B = np.array(B)
    [i_A,j_A,m] = np.shape(A)
    #m = B.shape()
    [i_B,j_B,m] = np.shape(B)
    output = np.ones([i_A-i_B+1,j_A-j_B+1])
    i_out = i_A-i_B+1
    j_out = j_A-j_B+1
    print(i_out,j_out)
    for i_index in range(0,i_out):
        for j_index in range(0,j_out):
            Tem = np.zeros([m,i_B,j_B])
            Tem = A[i_index:i_index+i_B,j_index:j_index+j_B,:]*B
            output[i_index,j_index] = sum(sum(sum(Tem)))
    return(output)


image_raw_data_jpg = tf.gfile.FastGFile('image3.jpg', 'rb').read()

with tf.Session() as sess:
        img_data_jpg = tf.image.decode_jpeg(image_raw_data_jpg) #图像解码
        #img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.uint8) #改变图像数据的类型

        print(np.shape(img_data_jpg.eval()))
        C = np.asarray(img_data_jpg.eval())
        
        B = np.zeros([3,3,3])
        B[:,:,1] = [
            [1,0,-1],
            [0,1,-1],
            [1,-1,1],
            ]
        B[:,:,2] = [
            [1,0,-1],
            [0,1,-1],
            [1,-1,1],
              ]
        B[:,:,0] = [
            [1,0,-1],
            [0,1,-1],
            [1,-1,1],
            ]

        

        D = my_conv(C,B)
        
        plt.figure(1) #图像显示
        plt.imshow(img_data_jpg.eval())

        plt.figure(2)
        plt.imshow(D)

        plt.figure(3)
        plt.plot(D)
        plt.show()
        
    
