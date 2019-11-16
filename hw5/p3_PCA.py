import pandas as pd
import numpy as np
import scipy.io as sio
import scipy.stats
import matplotlib.pyplot as plt
import math
import time

yale = sio.loadmat('yalefaces.mat') 
yalefaces = yale['yalefaces']
d1, d2, n = yalefaces.shape

fig, ax = plt.subplots()
for i in range(0,yalefaces.shape[2]):
    x = yalefaces[:,:,i]
    ax.imshow(x, extent=[0, 1, 0, 1]) 
    plt.imshow(x, cmap=plt.get_cmap('gray'))
    #time.sleep(0.1) 
    plt.show()

def Vector(X):
    d1, d2, n = X.shape
    X = X.reshape((d1*d2,n))
    return X

def Center(X):
    p, n = X.shape
    X_sd = np.std(X, axis = 1)
    #identity = np.identity(p)
    column = np.ones(n).reshape((n,1))
    X_bar = np.mean(X, axis = 1).reshape((-1,1))
    X_center = X - X_bar.dot(column.T)
    return X_center

def SVD(X):
    V,D = np.linalg.eig(X.dot(X.T))
    Vsort = np.sort(V,axis = 0)
    Vsort[:] = Vsort[::-1]
    #print(Vsort.shape)
    VsortInd = np.argsort(V,axis =0) 
    VsortInd[:] = VsortInd[::-1]
    D = D[:,VsortInd] 
    V = np.copy(Vsort) 
    return V,D

def EvaluateComponents(V, percent):
    value = sum(V)
    kvalue = 0
    for i in range(0,V.shape[0]):
        kvalue = kvalue + V[i]
        if float(kvalue)/float(value) > percent:
            k = i+1
            break
    reduction = 1-(float(k)/float(V.shape[0]))
    return k, reduction

X = Center(Vector(yalefaces))
#print(X.shape) # (2016, 2414)
V,D = SVD(X)

plt.semilogy(V) 
plt.ylabel('log eigen values')
plt.xlabel('eigen values')
plt.show()

k1, reduction1 = EvaluateComponents(V, 0.95)
print(k1, reduction1)
k2, reduction2 = EvaluateComponents(V, 0.99)
print(k2, reduction2)

eig0 = np.reshape(np.mean(Vector(yalefaces),axis=1), [d1,d2])
f, axarr = plt.subplots(4, 5) 
for i in range(0,4):
    for j in range(0,5):
        if i == 0 and j ==0:
            axarr[i, j].imshow(eig0, cmap=plt.get_cmap('gray')) 
        else:
            px = np.reshape(D[:,i*5+j-1],[d1,d2])
            axarr[i, j].imshow(px, cmap=plt.get_cmap('gray'))
plt.show()