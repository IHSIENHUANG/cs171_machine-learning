
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import decimal  
import sys
import scipy
from scipy.stats.stats import pearsonr


# In[34]:



iris_data = pd.read_csv('iris.data',header=None)
wine_data = pd.read_csv('wine.data',header=None)
class_count_wine = [0,59,130,178]
class_count_iris = [0,50,100,150]
titles_iris = ["Iris-setosa","Iris-versicolor","Iris-virginica"]

#print (choose_feature[1])
titles_wine = ["WINEA","WINEB","WINEC"]
print ("please input ur choice 1 for iris 2 for wine")
choice = input()
if int(choice) ==1:#depends on the user's choice to decide the input data
    print ("data is iris")
    data = iris_data
    VAL = class_count_iris
    titles = titles_iris
    choose_feature = [0,4]
else:
    print ("data is wine")
    data = wine_data 
    VAL = class_count_iris
    titles = titles_wine
    choose_feature = [1,5]
data_class = []
for i in range(1,4):#Based on the input class to split the data
    data_class.append(data[VAL[i-1]:VAL[i]])
BINS = [5,10,50,100]
for class_num in range(0,3):#for different class 
    for index in range(choose_feature[0],choose_feature[1]):#for different features
        plt.figure(figsize=(20,20))
        for i,bins in enumerate(BINS): 
            iris_data = (data_class[class_num][index])
            min_num = min(iris_data)
            max_num = max(iris_data)
            print ("min max",min_num,max_num)
            block = ((max_num-min_num)/float(bins))
            print ("block size",block)
            max_label = max_num+block
            x_label = np.arange(min_num,max_label,block) 
            x = np.arange(min_num,max_num,block)#make sure the x size
            print("min max:",min_num,max_num)
            print(x)
            y = np.zeros(shape=(1,bins))#size of bins
            print (y)
            for val in iris_data:
                pos = int(round(val-min_num,4)/(block))
                if pos ==bins:
                    pos =bins-1
                y[0][pos] +=1  
            print(y[0])
            print("sum is :",sum(y[0]))
            plt.subplot(2,2,i+1)
            #plt.bar(np.arange(len(x)),y[0])
            
            plt.axis('on')
            plt.xticks(x_label,rotation='vertical')
            plt.bar(x+block/2,y[0],width = 0.25/bins)
            #plt.set_xl
            plt.title(titles[class_num]+"Feature:"+str(index))
            plt.tight_layout()
        plt.savefig(titles[class_num]+"Feature:"+str(index)+".png")
        plt.show()


# In[ ]:


plt.figure(figsize=(10,15))
for class_num in range(0,3):#for different class 
    for index in range(choose_feature[0],choose_feature[1]):#for different features
        plt.subplot(4,3,class_num*4+index+1-choose_feature[0])# wine featrues first one is 1, therefore, the func should minus 1
        plt.boxplot(data_class[class_num][index], 1)
        plt.title(titles[class_num]+" Feature: "+str(index))
plt.savefig("BOXPLOT_"+titles[class_num]+"Feature:"+str(index)+".png")
plt.show()



# In[5]:


def dist(data1,data2):
    res = scipy.stats.pearsonr(data1, data2)#this is the checking that my result is correct
    print ("real result",res)
    #print (data1,data2)
    distance = 0
    data1_std,data2_std =np.std(data1),np.std(data2)
    data1_mean,data2_mean  = np.mean(data1),np.mean(data2)
    for val1,val2 in zip(data1,data2):
        distance += (val1-data1_mean)*(val2-data2_mean)
    distance = distance/len(data1)/(data1_std*data2_std)
    print ("total distance :",distance)
    return distance
        


# In[12]:


def distance_heatmap():
    dist_f = dist(np.arange(0,10),np.arange(10,20))
    print (dist_f)
    print ("please input ur choice 1 for iris 2 for wine")
    choice = input()
    if int(choice) ==1:
        data =  pd.read_csv('iris.data',header=None)
        titles = 'iris_'
        choose_feature=[0,4]
    else:
        titles = 'wine_'
        data = pd.read_csv('wine.data',header=None)
        choose_feature=[1,14]
    matrix = []
    count = -1 
    for i in range(choose_feature[0],choose_feature[1]):
        count +=0
        matrix.append([])
        for j in range(choose_feature[0],choose_feature[1]):
            print ("---new round---")
            distance = dist(data[i],data[j])
            print ("i and j dis:",i,j,round(distance,4))
            matrix[count].append(round(distance,4))
    print (matrix)
    '''
    import seaborn as sns
    fig = plt.figure(figsize=(4,4))
    r = sns.heatmap(matrix, cmap='BuPu')
    '''
    ax = plt.subplot(111)
    cax = ax.matshow(matrix, interpolation='nearest')
    plt.colorbar(cax)
    plt.title(titles)
    plt.savefig(titles+"cor.png")
    plt.show()


# In[25]:


def scatter():#this one is to print iris data
    data =  pd.read_csv('iris.data',header=None)
    COLORS = ['red','green','blue']
    colors = []
    for i in range(0,150):
        if i<50:
            colors.append(COLORS[0])
        elif i <100:
            colors.append(COLORS[1])
        else:
            colors.append(COLORS[2])
    for i in range(0,4):
        plt.figure(figsize=(9,12))
        for j in range(i+1,4):
            plt.figure()
            plt.title("feature "+str(i)+" +feature "+str(j))
            plt.scatter(data[i], data[j], c=colors, alpha=0.5)
            plt.savefig("feature"+str(i)+"feature"+str(j)+".png")
            plt.show()
        


# In[37]:


import math
def distance_LP(x,y,p):
    #print("this function is for calculating distance_LP")
    distance=0
    for val1,val2 in zip(x,y):
        #print ("value 1,2 :",val1,val2)
        distance += math.pow(abs(val1-val2),p)
        #print ("distance sum is ",distance)
    return distance
def Data_Lp_Norm():
    print ("this is for two data sets LP norm")
    print ("please input ur choice 1 for iris 2 for wine")
    choice = input()
    if int(choice) ==1:#depends on the user's choice to decide the input data
        print ("data is iris")
        data = pd.read_csv('iris.data',header=None)
        choose_feature = [0,4]#it means 0 to 3
        titles = "iris"
        label_index = 4
    else:
        print ("data is wine")
        data = pd.read_csv('wine.data',header=None)
        choose_feature = [1,14]#it means 1 to 13
        titles = "wine"
        label_index =0
    print ("lenght of data is ",len(data))#remember to skip the label
    
    p = 1
    index_start = int(choose_feature[0])
    index_end = int(choose_feature[1])
    for tmp in range(1,3):
        p = tmp
        plt.figure(figsize=(10,10))
        matrix = []
        #print (data[4])
        total_get = 0
        for i, x in enumerate(data.values):
            min_distance = 100
            label_x = x[label_index]
            matrix.append([])
            #print ("i and val is ",i,val[0:4])
            x = x[index_start:index_end]
            for j,y in enumerate(data.values):
                
                tmp_label = y[label_index]
                
                y = y[index_start:index_end]
                 
                dist = distance_LP(x,y,p)
                if dist < min_distance and i !=j: # we can skip the distance of itself
                    min_distance = dist
                    label_y = tmp_label
                matrix[i].append(dist)
            if str(label_x) == str(label_y):
                total_get+=1
        print ("total point is" ,total_get)
        
        ax = plt.subplot(111)
        cax = ax.matshow(matrix, interpolation='nearest')
        plt.colorbar(cax)
        plt.title(titles+str(p))
        plt.savefig(titles+str(p)+".png")
        plt.show()
    
        


# In[40]:


def main():

    #distance_heatmap()
    #scatter()#this one if for scatter
    #x = [1,2,3,4]
    #y = [4,3,3,0]
    Data_Lp_Norm()

if __name__ == "__main__":
    main()

