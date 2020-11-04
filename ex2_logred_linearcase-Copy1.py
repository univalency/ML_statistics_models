#!/usr/bin/env python
# coding: utf-8

# In[11]:


plt.scatter(X_norm[pos[:,0],1],X_norm[pos[:,0],2],c="r",marker="+",label="Admitted")
plt.scatter(X_norm[neg[:,0],1],X_norm[neg[:,0],2],c="b",marker="x",label="Not admitted")
x_value= np.array([np.min(X_norm[:,1]),np.max(X_norm[:,1])])
y_value=-(theta[0] +theta[1]*x_value)/theta[2]
plt.plot(x_value,y_value, "r")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend(loc=0)


# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
#print (os.listdir("../Andrew_Ng_ML"))
data = pd.read_csv('../Andrew_Ng_ML/ex2_data1.txt', header = None)


X = data.iloc[:,0:2].values
y = data.iloc[:,2].values
x1 = data.iloc[:,0]
x2 = data.iloc[:,1]
y = y[:, np.newaxis]
alpha = 0.01
iterations = 1500
m = len(y)
theta = np.zeros((X.shape[1] + 1,1))
ones = np.ones((m,))
print (x1.shape , theta.shape , ones.shape)

pos , neg = (y==1).reshape(m,1) , (y==0).reshape(m,1)
plt.figure(figsize = (12,8))
plt.scatter(X[pos[:,0],0],X[pos[:,0],1],c="r",marker="+")
plt.scatter(X[neg[:,0],0],X[neg[:,0],1],marker="o",s=10)
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend(["Admitted","Not admitted"],loc=0)


# In[13]:


def sigmoid (z):
    return (1/(1+np.exp(-z)))
print (sigmoid(0))


# In[14]:


def Cost (X, y, theta):
    predictions = sigmoid(np.matmul(X,theta))
    cost = (1/m)*(np.dot(-y.T,np.log(predictions)) - np.dot(np.add(ones, -y).T, np.log(np.add(ones , -predictions))))
    grad = (1/m)*np.matmul(X.T , (predictions -y))
    return grad , cost[0][0]
grad, cost= Cost(X,y,np.zeros((X.shape[1],1)))
print (cost)


# In[15]:


def featureNormalize(X):
    mean = np.mean(X, axis = 0)
    std = np.std(X,axis = 0)
    X_norm = (X - mean)/std
    return X_norm, mean, std
X_norm , mean, std = featureNormalize(X)


# In[19]:


iterations = 400
alpha = 1
def GradientDescent(X , y, theta):
    J_history = []
    for _ in range(iterations):
        grad, cost = Cost(X,y,theta)
        theta = theta - (alpha * grad)
        J_history.append(cost)
    return theta , J_history
x1_norm , x1meean, x1std = featureNormalize(x1)
x2_norm , x2meean, x2std = featureNormalize(x2)
X_norm = np.stack((ones, x1_norm, x2_norm), axis = 1)
theta , J_history = GradientDescent(X_norm , y ,theta)
print (J_history[0],GradientDescent(X_norm , y ,theta)[0])


# In[20]:


plt.figure(figsize = (12,8))
plt.scatter(np.array(list(range(0, iterations))),np.array(J_history))



# In[21]:



x_test = np.array([45,85])
x_test = (x_test - mean)/std
x_test = np.append(np.ones(1),x_test)
prob = sigmoid(x_test.dot(theta))
print("For a student with scores 45 and 85, we predict an admission probability of",prob[0])


# In[ ]:





# In[ ]:




