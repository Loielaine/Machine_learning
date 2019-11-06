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
plt.scatter(x=data[:,0],y=data[:,1],c=LDA_result)
plt.xlabel('data[:,0]')
plt.ylabel('data[:,1]')
plt.title('LDA')
plt.show()

# QDA plot
plt.scatter(x=data[:,0],y=data[:,1],c=QDA_result)
plt.xlabel('data[:,0]')
plt.ylabel('data[:,1]')
plt.title('QDA')
plt.show()