import pandas as pd
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def SolvePenalizedLinearRegression(X, y, c):
    p,n = X.shape 
    X0 = np.ones((1,n))
    Xnew = np.vstack((X0,X))
    xy = np.dot(Xnew, y) #(p+1)*1
    xx = np.dot(Xnew, np.transpose(Xnew))
    identity = np.identity(p+1)
    identity[0,0] = 0
    mat = identity * c + xx #(p+1)*(p+1)
    matinv = np.linalg.inv(mat) #(p+1)*(p+1)
    w = np.dot(matinv, xy) 
    return w

def CalculateMeanSquareError(X, y, w):
    p,n = X.shape 
    X0 = np.ones((1,n))
    Xnew = np.vstack((X0,X))
    error = np.dot(np.transpose(Xnew),w) - y
    MSE = np.mean(np.sum(error**2))/n
    return MSE


bodyfat = scipy.io.loadmat('bodyfat_data.mat')
X = bodyfat['X'].T
X_train = X[:,0:149]
X_test = X[:,150:]
y = bodyfat['y']
y_train = y[0:149,:]
y_test = y[150:,:]


w = SolvePenalizedLinearRegression(X_train, y_train, 10)
print(w)

MSE = CalculateMeanSquareError(X_test , y_test, w)
print(MSE)