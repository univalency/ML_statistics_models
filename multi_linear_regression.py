#!/usr/bin/env python
# coding: utf-8

# In[81]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
print (os.listdir("../Andrew_Ng_ML"))
data = pd.read_csv('../Andrew_Ng_ML/ex1_data2.txt', header = None)


# In[82]:


x1 = data.iloc[:,0]
x2 = data.iloc[:,1]
y = data.iloc[:,2]
y = y[:, np.newaxis]
theta = np.zeros([3,1])
alpha = 0.01
iterations = 1500
ones = np.ones((m,))
m = len(y)
data.head()
print (y.shape)


# In[83]:


def featureNormalize(X):
    mu = np.zeros([1,m])
    sigma = np.zeros([1,m])
    mu = (1/m)*np.sum(X)
    X_norm = X - mu
    sigma = np.std(X)
    X_norm = (1/sigma) * X_norm 
    return X_norm
print (featureNormalize(x1).shape)


# In[84]:


X = np.stack((ones, featureNormalize(x1), featureNormalize(x2)), axis = 1)
print (X.shape)


# In[85]:


def computeCost(X, y, theta):
    hypothesis = np.matmul(X , theta) - y
    return np.sum(np.power(hypothesis, 2))/ (2*m)
J = computeCost(X,y,theta)
print (J)


# In[86]:


def gradientDescent(X, y, theta, alpha, iterations):
    for _ in range(iterations):
        temp = np.matmul(X , theta)-y
        temp = np.matmul(X.T, temp)
        theta = theta - (alpha/m) * temp
    return theta
theta = gradientDescent(X, y, theta, alpha, iterations)
print (theta)


# In[ ]:




