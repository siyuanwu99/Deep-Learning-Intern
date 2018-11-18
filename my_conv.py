#!usr/bin/env python

import numpy as np
input_data=[
            [10,10,10,0,0,0],
            [10,10,10,0,0,0],
            [10,10,10,0,0,0],
            [0,0,0,10,10,10],
            [0,0,0,10,10,10],
            [0,0,0,10,10,10]
            ]
weights_data=[ 
               [ 1, 1, 1],
                [0, 0, 0],
                [ -1,-1,-1],
           ]



def my_conv(A,B):
    '''
    convolution
    Argument:
    A -- Matrix
    B -- filter
    
    Output:
    output -- matrix

    
    '''
    A = np.array(A)
    B = np.array(B)
    [i_A,j_A] = np.shape(A)
    #m = B.shape()公式公式
    [i_B,j_B] = np.shape(B)
    output = np.ones([i_A-i_B+1,j_A-j_B+1])
    i_out = i_A-i_B+1
    j_out = j_A-j_B+1
    print(i_out,j_out)
    for i_index in range(0,i_out):
        for j_index in range(0,j_out):
            Tem = np.zeros([i_B,j_B])
            Tem = A[i_index:i_index+i_B,j_index:j_index+j_B]*B
            output[i_index,j_index] = sum(sum(Tem))
    return(output)

output_data = my_conv(input_data,weights_data)

print(output_data)
