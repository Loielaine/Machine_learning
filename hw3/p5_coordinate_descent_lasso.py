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

def CoordinateGradient(X, y, w,i):
    p,n = X.shape
    X0 = np.ones((1,n))
    Xnew = np.vstack((X0,X))
    Xi = Xnew[i,:].reshape((1,n))
    w0 = np.mean(y)
    w[0] = w0
    w_i = np.copy(w)
    w_i[i] = 0
    y_wi = Xnew.T.dot(w_i).reshape((n,1)) 
    y = y.reshape((n,1))
    #print(Xi.shape, y.shape, y_wi.shape)
    a = 2 * Xi.dot(Xi.T).reshape(-1)
    c = 2 * Xi.dot(y - y_wi).reshape(-1)
    #print(a.shape, c.shape)
    return a , c

def CoordinateDescent(X,y,w,it,lamda):
    p,n = X.shape 
    w_update = w 
    w_it = np.zeros(shape = (p+1,it))
    MSE_it = []
    for it in range(it):
        for i in range(1,p+1):
            a,c = CoordinateGradient(X, y, w, i)
            if(c>lamda):
                w_update[i] = (c-lamda)/a
            elif(c<-lamda):
                w_update[i] = (c+lamda)/a
            else:
                w_update[i] = 0        
            w = w_update
        #print(w.shape)
        w_it[:,it] = w.reshape(-1)
        error = CalculateMeanSquareError(X, y, w)
        MSE_it.append(error)
        #print(w_it.shape)
    return w_it, MSE_it
  
def CalculateMeanSquareError(X, y, w):
    p,n = X.shape 
    y = y.reshape((n,1))
    X0 = np.ones((1,n))
    Xnew = np.vstack((X0,X))
    error = np.sum((y - Xnew.T.dot(w)).T.dot(y - Xnew.T.dot(w)))/n
    return error


def CrossValidationCoordinateDescent(X , y, folds ,w,it):
    p,n = X.shape 
    MSE_it = []
    w_it = np.zeros(shape = (p+1,1000))
    for i in range(1000):
        lamda = i/10.
        MSE = []
        for k in range(0,folds):
            X_valid = X[:, k*46 :  (k+1)*46].reshape((p,-1))
            y_valid = y[k*46 :  (k+1)*46].reshape((-1,1))
            index = X == X
            index[ :, k*46 :  (k+1)*46] = False
            X_train = X[index].reshape((p,-1))
            index  = y == y
            index[k*46 :  (k+1)*46] = False
            y_train = y[index].reshape((-1,1))

            w_kit, MSE_kit = CoordinateDescent(X_train,y_train,w,it,lamda)
            w_k = w_kit[:,-1].reshape((14,1))
            MSE_k = CalculateMeanSquareError(X_valid, y_valid, w_k)
            MSE.append(MSE_k)
        cv_MSE = np.mean(MSE)
    #cv_df = CalculateEffectiveDF(X,w)
        cv_wit, cv_MSEit = CoordinateDescent(X,y,w,it,lamda)
        w_it[:,i] = cv_wit[:,-1].reshape(-1)
        MSE_it.append(cv_MSE)
    return w_it, MSE_it


X_test = X[:,-46:]
y_test = y[-46:]
X_cv = X[:,:-46]
y_cv = y[:-46]
X_cv , X_bar, X_sd = Sphere(X_cv)

w0 = np.ones((14,1))
it = 50
lamda = 100
w_it, MSE_it = CoordinateDescent(X_cv,y_cv,w0,it,lamda)

print("The weights of CD-lasso regression are ",w_it[:,-1].reshape((14,1)))

train_error = MSE_it[-1]
test_error = CalculateMeanSquareError(SphereTestSet(X_test,X_bar, X_sd), y_test, w_it[:,-1].reshape((14,1)))
print("The train MSE of CD-lasso regression is ", train_error)
print("The test MSE of CD-lasso regression is ", test_error)

for i in range(1,14):
    wi = w_it[i,:]
    plt.plot(range(it),  wi)
    plt.xlabel('iteration')
    plt.ylabel('weights')
plt.show()

plt.plot(range(it),  MSE_it)
plt.xlabel('iteration')
plt.ylabel('MSE for training data')
plt.show()

# cross-validation
w0 = np.ones((14,1))
it = 50
w_it, MSE_it = CrossValidationCoordinateDescent(X_cv,y_cv,10,w0,it)


lambda_list = np.arange(0,100,0.1)
plt.plot(lambda_list,  MSE_it)
plt.xlabel('lambda')
plt.ylabel('MSE for cross validation')
plt.show()

best_lamda = range(1000)[np.argmin(MSE_it)]/10
print("The best lambda is ", best_lamda)


