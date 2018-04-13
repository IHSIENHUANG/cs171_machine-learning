
# coding: utf-8

# In[82]:


import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import decimal  


# In[94]:


val = 5.1
print(val)
val = round(val,4)
min_num =4.3
min_num =round(min_num,4)
print (round(val -min_num,4))


# In[111]:


iris_data = pd.read_csv('iris.data',header=None)
iris_class = []
iris_features = []
iris_class.append(iris_data[0:50])
iris_class.append(iris_data[50:100])
iris_class.append(iris_data[100:150])
titles = ["Iris-setosa","Iris-versicolor","Iris-virginica"]
bins = 5
for class_num in range(0,3):
    for index in range(0,4):
        iris_data = (iris_class[class_num][index])
        min_num = min(iris_data)
        max_num = max(iris_data)
        block = (max_num-min_num)/(bins)
        print ("block size",block)
        x = np.arange(min_num,max_num,block)#make sure the x size
        print("min max:",min_num,max_num)
        print(x)
        y = np.zeros(shape=(1,bins))#size of bins
        print (y)
        for val in iris_data:
            pos = int(round(val-min_num,4)/(block))
            #print ("val:",val)
            #print ("val - min :",round(val-min_num,4))
            #print ("min_num :",min_num)
            if pos ==bins:
                pos =bins-1
            #print ("pos:",pos)
            y[0][pos] +=1  
        print(y[0])
        plt.figure(figsize=(9,6))
        plt.subplot(2,2,1)
        plt.bar(range(len(x)),y[0],width=0.1,facecolor="lightskyblue",lw=1,edgecolor='white')
        plt.title("class: "+titles[class_num]+" Feature: "+str(index))
    plt.show()


# In[74]:




