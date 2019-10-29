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

def SphereTestSet(X,X_bar, X_sd):
    p, n = X.shape
    column = np.ones(n).reshape((n,1)) 
    return np.diag(1/X_sd).dot(X - X_bar.dot(column.T))

def SolveRidgeRegression(X, y, c):
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

def CalculateEffectiveDF(X,w):
    p,n = X.shape 
    xx = np.dot(X, np.transpose(X))
    identity = np.identity(p)
    mat = identity * c + xx 
    matinv = np.linalg.inv(mat) 
    df_mat =  X.T.dot(matinv).dot(X)
    df = np.sum(np.diagonal(df_mat))
    return df

def CalculateMeanSquareError(X, y, w):
    p,n = X.shape 
    y = y.reshape((n,1))
    w = w.reshape((p+1,1))
    X0 = np.ones((1,n))
    Xnew = np.vstack((X0,X))
    error = np.sum((y - Xnew.T.dot(w)).T.dot(y - Xnew.T.dot(w)))/n
    return error


def CrossValidation(X , y, c, folds):
    p,n = X.shape 
    fold_size = [46, 46, 46, 46, 46, 46, 46, 46, 46, 46]
    MSE = []
    df= []
    for k in range(0,folds):
        print(k*46,(k+1)*46)
        X_valid = X[:, k*46 :  (k+1)*46]
        y_valid = y[k*46 :  (k+1)*46]
        #print(X_valid.shape)
        #print(y_valid.shape)
        index = X == X
        index[ :, k*46 :  (k+1)*46] = False
        X_train = X[index].reshape((p,-1))
        
        index  = y == y
        index[k*46 :  (k+1)*46] = False
        y_train = y[index].reshape((-1,1))
        
        
        w_k = SolveRidgeRegression(X_train, y_train, c)
        MSE_k = CalculateMeanSquareError(X_valid, y_valid, w_k)
        MSE.append(MSE_k)
    cv_MSE = np.mean(MSE)
    cv_df = CalculateEffectiveDF(X,w)
    cv_w = SolveRidgeRegression(X, y, c)
    #print(cv_w.shape)
    return cv_MSE, cv_df,cv_w

X_test = X[:,-46:]
y_test = y[-46:]
X_cv = X[:,:-46]
y_cv = y[:-46]
X_cv , X_bar, X_sd = Sphere(X_cv)

# ridge regression
MSE_split = []
df_split = []
for c in np.arange(0, 20, 0.1):   
    w = SolveRidgeRegression(X_cv, y_cv, c)
    MSE_c = CalculateMeanSquareError(SphereTestSet(X_test,X_bar, X_sd), y_test, w)
    MSE_split.append(MSE_c )
    df = CalculateEffectiveDF(X_cv,w)
    df_split.append(df)

plt.plot(np.arange(0, 20, 0.1), df_split)
plt.xlabel('lambda')
plt.ylabel('Effective df')
plt.show()

# cross-validation
MSE_cv = []
df_cv = []
w_cv = []
for i in range(0,200):  
    c = i/10
    MSE_c, df_c ,w_c = CrossValidation(X_cv , y_cv, c, 10)
    MSE_cv.append(MSE_c)
    df_cv.append(df_c)
    w_cv.append(w_c)

plt.plot(np.arange(0, 20, 0.1),  MSE_cv)
plt.xlabel('lambda')
plt.ylabel('MSE of cross validation')
plt.show()