#!/usr/bin/env python
# coding: utf-8

# In[116]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
print (os.listdir("../Andrew_Ng_ML"))
data = pd.read_csv('../Andrew_Ng_ML/ex1_data1.txt', header = None)


# In[117]:


x = data.iloc[:,0]
print (x.shape)
y = data.iloc[:,1]
print (y.shape)


# In[118]:


m = len(y)
print (m)
data.head()


# In[119]:


plt.scatter(x, y)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()


# In[120]:


x = x[:,np.newaxis]
print (x.shape)
y = y[:,np.newaxis]
iterations = 1500
theta = np.zeros([2,1])
alpha = 0.01
ones = np.ones((m,1))
print (x.shape)
X = np.hstack((ones, x))
print (X.shape)


# In[121]:


def computeCost(X, y, theta):
    hypothesis = np.matmul(X , theta) - y
    return np.sum(np.power(hypothesis, 2))/ (2*m)
J = computeCost(X,y,theta)
print (J)
    


# In[122]:


temp = np.matmul(X , theta)-y 
print (theta)
print ('temp is' , temp[0:5])
temp = np.matmul(X.T, temp)
print (X[0:5])
print ('now it is' , temp)
theta = theta - (alpha/m) * temp


# In[123]:


def gradientDescent(X, y, theta, alpha, iterations):
    for _ in range(iterations):
        temp = np.matmul(X , theta)-y
        temp = np.matmul(X.T, temp)
        theta = theta - (alpha/m) * temp
    return theta
theta = gradientDescent(X, y, theta, alpha, iterations)
print (theta)


# In[125]:


J = computeCost(X, y, theta)
print (J)


# In[ ]:




