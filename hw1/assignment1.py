
# coding: utf-8

# In[97]:


import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import decimal  
import sys
import scipy
from scipy.stats.stats import pearsonr


# In[62]:



iris_data = pd.read_csv('iris.data',header=None)
wine_data = pd.read_csv('wine.data',header=None)
class_count_wine = [0,59,130,178]
class_count_iris = [0,50,100,150]
titles_iris = ["Iris-setosa","Iris-versicolor","Iris-virginica"]

print (choose_feature[1])
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
        plt.figure()
        for i,bins in enumerate(BINS): 
            iris_data = (data_class[class_num][index])
            min_num = min(iris_data)
            max_num = max(iris_data)
            print ("min max",min_num,max_num)
            block = ((max_num-min_num)/float(bins))
            print ("block size",block)
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
            plt.bar(np.arange(len(x)),y[0])
            plt.title(titles[class_num]+" Feature: "+str(index))
            plt.tight_layout()
        plt.show()


# In[64]:


plt.figure(figsize=(10,15))
for class_num in range(0,3):#for different class 
    for index in range(choose_feature[0],choose_feature[1]):#for different features
        plt.subplot(4,3,class_num*4+index+1-choose_feature[0])# wine featrues first one is 1, therefore, the func should minus 1
        plt.boxplot(data_class[class_num][index], 1)
        plt.title(titles[class_num]+" Feature: "+str(index))
plt.savefig(titles[class_num]+"Feature:"+str(index)+".png")
plt.show()



# In[114]:


def dist(data1,data2):
    res = scipy.stats.pearsonr(data1, data2)
    print ("real result",res)
    print (data1,data2)
    distance = 0
    data1_std,data2_std =np.std(data1),np.std(data2)
    data1_mean,data2_mean  = np.mean(data1),np.mean(data2)
    for val1,val2 in zip(data1,data2):
        distance += (val1-data1_mean)*(val2-data2_mean)
    distance = distance/len(data1)/(data1_std*data2_std)
    print ("total distance :",distance)
    return distance
        


# In[115]:


dist_f = dist(np.arange(0,10),np.arange(10,20))
print (dist_f)
from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#def Person_cor():
rs = np.random.RandomState(33)
d = pd.DataFrame(data=rs.normal(size=(100, 26)),columns=list(ascii_letters[26:]))

# Compute the correlation matrix
corr = d.corr()
print (corr)


# In[116]:


def bar_data():
    print ("this is part1")
def main():
    Person_cor()
    bar_data()
if __name__ == "__main__":
    main()

