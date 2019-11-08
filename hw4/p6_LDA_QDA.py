import pandas as pd
import numpy as np
import numpy.random as r
import scipy.io as sio
import scipy.stats
import matplotlib.pyplot as plt
import random
from collections import Counter
pi = np.pi

def LDA(mu0, mu1, cov,X):
    return ((mu1-mu0).T).dot(np.linalg.inv(cov)).dot(X) - 1/2*((mu1.T).dot(np.linalg.inv(cov)).dot(mu1)-(mu0.T).dot(np.linalg.inv(cov)).dot(mu0))

def QDA(mu0, mu1, cov0, cov1, X):
    term1 = 1/2*(((X-mu0).T).dot(np.linalg.inv(cov0)).dot((X-mu0))-((X-mu1).T).dot(np.linalg.inv(cov1)).dot((X-mu1)))
    #print(term1.shape)
    term2 = 1/2*math.log(np.linalg.det(cov0)/np.linalg.det(cov1))
    return term1+term2

from sklearn.datasets import load_iris
iris=load_iris()
# You have two features and two classifications
data_0, data_1 = iris.data[:,1:3][:50], iris.data[:,1:3][50:100]

mu0 = np.mean(data_0,axis=0).reshape((2,1))
mu1 = np.mean(data_1,axis=0).reshape((2,1))
cov = np.cov(data,rowvar=False).reshape((2,2))
cov0 = np.cov(data_0,rowvar=False).reshape((2,2))
cov1 = np.cov(data_1,rowvar=False).reshape((2,2))

LDA_score = LDA(mu0, mu1, cov, data.T).reshape(-1)
LDA_result = np.array(LDA_score>0)
QDA_score = np.diag(QDA(mu0, mu1, cov0, cov1, data.T))
QDA_result = np.array(QDA_score>0)

# LDA plot
w = (mu1-mu0).T.dot(np.linalg.inv(cov))
b = -1/2*((mu1.T).dot(np.linalg.inv(cov)).dot(mu1)-(mu0.T).dot(np.linalg.inv(cov)).dot(mu0))[0][0]
x = np.arange(0,6,0.1)
y = (w[0,0]*x+b)/(-w[0,1])
plt.scatter(x=data[:,0],y=data[:,1],c= LDA_result.reshape(-1))
plt.plot(x,y)
plt.xlabel('data[:,0]')
plt.ylabel('data[:,1]')
plt.title('LDA')
plt.show()

# QDA plot
from sympy import plot_implicit, cos, sin, symbols, Eq, And

x, y = symbols('x y')
X = np.array([x,y]).reshape((2,1))
diff1 = np.linalg.inv(cov0) - np.linalg.inv(cov1)
diff2 = np.linalg.inv(cov0).dot(mu0) -  np.linalg.inv(cov1).dot(mu1)
threshold = 1/2*((mu1.T).dot(np.linalg.inv(cov)).dot(mu1)-(mu0.T).dot(np.linalg.inv(cov)).dot(mu0))[0][0] + \
1/2*np.log(np.linalg.det(cov1)/np.linalg.det(cov0))
expr = 1/2*(X.T.dot(diff1).dot(X)) - X.T.dot(diff2) - threshold

plt2 = plot_implicit(Eq(expr[0,0],0),(x,-5,5), (y,-6,6))
plt2._backend.ax.scatter(data_0[:,0],data_0[:,1],label='data_0')
plt2._backend.ax.scatter(data_1[:,0],data_1[:,1],label='data_1')
plt2._backend.save('plt2.png')