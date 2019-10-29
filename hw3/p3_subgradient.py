import pandas as pd
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random

np.random.seed(0)
nuclear = sio.loadmat('nuclear.mat')
x = nuclear['x']
y = nuclear['y']

def CalculateThetaTerm(x, y ,theta,n):
    ThetaTerm = np.concatenate([np.zeros([1,n]),1-y*(theta[0:-1].dot(x) + theta[-1])])
    return ThetaTerm

def SubGradientDescent(w,b,x,y,lamda,n):
    #y = y.reshape((-1,1))
    #b = b.reshape(-1)
    #print(b.shape)
    #print(y.shape)
    if ((1.0 - y*(w.dot(x) + b)) > 0):
        u = np.concatenate([-1./n * (y*x  - lamda*w), np.array([-1./n*y])],axis = 0)
    else:
        u = np.concatenate([1./n * lamda*w, np.array([0])],axis = 0)
    return u

n = X.shape[1]
negInd = y==-1
posInd = y==1
plt.scatter(x[0,negInd[0,:]], x[1,negInd[0,:]],color='b')
plt.scatter(x[0,posInd[0,:]], x[1,posInd[0,:]],color='r')
plt.figure(1)
plt.show()

# Subgradient
np.random.seed(0)
theta = np.array([0,0,0])
lamda = 0.001
it = 40
p, n = x.shape
ThetaTerm = CalculateThetaTerm(x, y ,theta,n)
obj = (1. / n) * np.sum(np.max(ThetaTerm,axis = 0))  + (lamda/2) *(np.linalg.norm(theta[0:-1],2) ** 2 )
obj_plot = np.zeros([it])
for j in range(0,it):
    theta_old = np.copy(theta)
    u = np.array([0,0,0])
    for i in range(0,n):
        u =  u +  SubGradientDescent(theta[0:-1], theta[-1],x[:,i],y[0,i],lamda,n)
    theta = theta - (100./(j+1.)) * u
    ThetaTerm = CalculateThetaTerm(x, y,theta,n)
    new_obj = (1. / n) * np.sum(np.max(ThetaTerm,axis = 0))  + (lamda/2) *(np.linalg.norm(theta[0:-1],2) ** 2 )
    obj_plot[j] = new_obj

plt.plot(range(it),  obj_plot)
plt.xlabel('iteration')
plt.ylabel('Loss')
plt.show()

# Stocastic subgradient
np.random.seed(0)
theta = np.array([0,0,0])
p, n = x.shape
ThetaTerm = CalculateThetaTerm(x, y ,theta,n)
obj = (1. / n) * np.sum(np.max(ThetaTerm,axis = 0))  + (lamda/2) *(np.linalg.norm(theta[0:-1],2) ** 2 )
it = 10
obj_plot =[]
for j in range(0,it):
    for i in range(0,n):
        theta_old = np.copy(theta)
        u =  SubGradientDescent(theta[0:-1], theta[-1],x[:,i],y[0,i],lamda,n)
        theta = theta - (100./(j+1.)) * u
        ThetaTerm = np.concatenate([np.zeros([1,n]),1-y*(theta[0:-1].dot(x) + theta[-1])])
        new_obj = (1. / n) * np.sum(np.max(ThetaTerm,axis = 0))  + (lamda/2) *(np.linalg.norm(theta[0:-1],2) ** 2 )
        obj_plot.append(new_obj)

plt.plot(range(it*n),  obj_plot)
plt.xlabel('iteration')
plt.ylabel('Loss')
plt.show()

