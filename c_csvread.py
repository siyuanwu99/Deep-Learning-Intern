import csv
import numpy as np 

def my_read( str1 ):
    x_data=[]
    y_data=[]            #定义空列表
    csvFile = open(str1, "r")
    reader = csv.reader(csvFile)
    for item in reader:
        # if reader.line_num==1:
            # continue
        item=[float(ii) for ii in item]
        x_data.append(item)

    ###把读取的数据转化成float格式

    for i in range(len(x_data)):
        y_data.append(x_data[i].pop())
    
    x_data = np.array(x_data)
    y_data = np.array(y_data)[:,np.newaxis]
    #y_data = np.transpose(y_data)   
    
    return x_data,y_data
