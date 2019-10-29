import pandas as pd
import numpy as np
import scipy.io
import matplotlib.pyplot as plt


def SplitXy(df):
    df_encode = pd.get_dummies(df,prefix=['species'])
    X = df_encode[['sepal_length', 'sepal_width', 'petal_length','petal_width']].values.T
    Y = df_encode[['species_setosa', 'species_versicolor','species_virginica']].values.reshape((-1,3))
    return X, Y
    
def FindKNearestNeighbors(X,Y,x,k):
    n,c = Y.shape 
    Xnew = np.hstack((X,x.reshape((-1,1))))
    #print(Xnew.shape)
    y0 = np.zeros((1,c))
    Ynew = np.vstack((Y, y0))
    #print(Ynew.shape)
    xx = np.dot(np.transpose(Xnew), Xnew)
    d = np.diag(xx)
    #print(d)
    d_x = np.tile(d[n],n+1)
    D = d_x - 2*xx[:,n] + d
    index = np.argsort(D,axis=0).reshape((-1,1))
    Y_sort = np.take_along_axis(Ynew,index,axis=0)
    #print(Y_sort)
    KNN = Y_sort[1:k+1,:]
    vote = np.sum(KNN,axis=0)
    #print(vote)
    majority = np.argmax(vote)
    return majority+1

def ObtainMisclassificationRate(y_predicted, y_test_encode):
    rate = float(sum(y_predicted == y_test_encode))/float(len(y_test_encode))
    return rate


train = pd.read_csv('iris.train.csv')
test = pd.read_csv('iris.test.csv')


X_train, Y_train = SplitXy(train)
X_test, Y_test = SplitXy(test)


Y_test[:,0] = Y_test[:,0] * 1
Y_test[:,1] = Y_test[:,1] * 2
Y_test[:,2] = Y_test[:,2] * 3
Y_test_encode = np.sum(Y_test, axis = 1)


p,n = X_test.shape
accuracy = []
for k in range(1,51):
    y_predicted = []
    for i in range(0,n):
        majority = FindKNearestNeighbors(X_train, Y_train, X_test[:,i],k)
        y_predicted.append(majority)
    rate = ObtainMisclassificationRate(y_predicted, Y_test_encode)
    accuracy.append(rate)


plt.plot(range(1,51), accuracy)
plt.show()