import pandas as pd
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random

house = pd.read_csv("boston-corrected.csv")
y = house['CMEDV'].values
X = house.loc[:,'CRIM':'LSTAT'].values.T

def Sphere(X):
    p, n = X.shape
    X_sd = np.std(X, axis = 1)
    #identity = np.identity(p)
    column = np.ones(n).reshape((n,1))
    X_bar = np.mean(X, axis = 1).reshape((-1,1))
    X_sphere = np.diag(1/X_sd).dot(X - X_bar.dot(column.T))
    print("The std is: ",X_sd)
    print("The mean is ",X_bar)
    return X_sphere,X_bar, X_sd

def Gradient(X, y, w):
    p,n = X.shape 
    X0 = np.ones((1,n))
    Xnew = np.vstack((X0,X))
    y = y.reshape((n,1))
    xy = np.dot(Xnew, y).reshape((p+1,1)) #(p+1)*1
    xx = np.dot(Xnew, np.transpose(Xnew)).reshape((p+1,p+1))
    g =  2/n * (xx.dot(w) - xy)
    #print(g.shape)
    return g

def GradientDescent(X,y,eta,w,it):
    p,n = X.shape 
    X0 = np.ones((1,n))
    Xnew = np.vstack((X0,X))
    #obj = np.mean((y - Xnew.T.dot(w))**2)
    w_update = w 
    w_it = np.zeros(shape = (p+1,it))
    MSE_it = []
    for i in range(it):
        w_update = w - eta * Gradient(X, y, w) 
        w_it[:,i] = w_update.reshape(-1)
        w = w_update
        error = CalculateMeanSquareError(X, y, w)
        MSE_it.append(error)
    return w_it.reshape((p+1,it)) , MSE_it
  
def CalculateMeanSquareError(X, y, w):
    p,n = X.shape 
    y = y.reshape((n,1))
    X0 = np.ones((1,n))
    Xnew = np.vstack((X0,X))
    error = np.sum((y - Xnew.T.dot(w)).T.dot(y - Xnew.T.dot(w)))/n
    return error

def SolveOLS(X,y):
    p,n = X.shape 
    X0 = np.ones((1,n))
    Xnew = np.vstack((X0,X))
    xy = np.dot(Xnew, y) #(p+1)*1
    xx = np.dot(Xnew, np.transpose(Xnew))
    matinv = np.linalg.inv(xx) #(p+1)*(p+1)
    w = np.dot(matinv, xy) 
    return w

X_test = X[:,-46:]
y_test = y[-46:]
X_cv = X[:,:-46]
y_cv = y[:-46]
X_cv , X_bar, X_sd = Sphere(X_cv)

eta = 0.05
w0 = np.zeros((14,1))
it = 500
w_it,MSE_it = GradientDescent(X_cv, y_cv,eta,w0,it)
w_GD = w_it[:,-1].reshape((14,1))

train_error = MSE_it[-1]
test_error = CalculateMeanSquareError(SphereTestSet(X_test,X_bar, X_sd), y_test, w_it[:,-1].reshape((14,1)))
print("The train MSE of GD regression is ", train_error)
print("The test MSE of GD regression is ", test_error)

plt.plot(range(it),  MSE_it)
plt.xlabel('iteration')
plt.ylabel('MSE for training data')
plt.show()

w_OLS = SolveOLS(X_cv, y_cv).reshape((14,1))

train_error = CalculateMeanSquareError(X_cv, y_cv,w_OLS)
test_error = CalculateMeanSquareError(SphereTestSet(X_test,X_bar, X_sd), y_test, w_OLS)
print("The train MSE of OLS regression is ", train_error)
print("The test MSE of OLS regression is ", test_error)
