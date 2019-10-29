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

def StocasticGradient(X, y, w):
    xy = np.dot(X, y).reshape((-1,1)) #(p+1)*1
    xx = np.dot(X, X.T)
    g =  2 * (xx.dot(w) - xy) 
    return g

def StocasticGradientDescent(X,y,eta,w,it):
    p,n = X.shape 
    X0 = np.ones((1,n))
    Xnew = np.vstack((X0,X))
    w_update = np.copy(w) 
    w_it = np.zeros(shape = (p+1,it))
    MSE_it = []
    for i in range(it):
        for j in range(1,n):
            #rand = random.randint(0,n-1)
            Xrand = Xnew[:,j].reshape((p+1,1))
            yrand = y[j]
            w_update = w - eta * StocasticGradient(Xrand, yrand, w) 
            w = np.copy(w_update) 
        w_it[:,i] = w.reshape(-1)
        error = CalculateMeanSquareError(X, y, w)
        MSE_it.append(error)
    return w_it.reshape((p+1,-1)), MSE_it
  
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

w0 = np.zeros((14,1))
eta = 0.0005
it = 500
w_it, MSE_it = StocasticGradientDescent(X_cv, y_cv, eta,w0,it)
w_SGD = w_it[:,-1].reshape((14,1))

train_error = MSE_it[-1]
test_error = CalculateMeanSquareError(SphereTestSet(X_test,X_bar, X_sd), y_test, w_SGD)
print("The train MSE of SGD regression is ", train_error)
print("The test MSE of SGD regression is ", test_error)

plt.plot(range(it),  MSE_it)
plt.xlabel('iteration')
plt.ylabel('MSE for training data')
plt.show()

w_OLS = SolveOLS(X_cv, y_cv).reshape((14,1))

train_error = CalculateMeanSquareError(X_cv, y_cv,w_OLS)
test_error = CalculateMeanSquareError(SphereTestSet(X_test,X_bar, X_sd), y_test, w_OLS)
print("The train MSE of OLS regression is ", train_error)
print("The test MSE of OLS regression is ", test_error)