import pandas as pd
import numpy as np
import numpy.random as r
import scipy.io as sio
import scipy.stats
import matplotlib.pyplot as plt
import random
from collections import Counter
pi = np.pi

np.random.seed(0)

def Kernel(x1, x2, sigma):
    x1 = x1.reshape((-1,1)) #d1*1
    x2 = x2.reshape((-1,1)) #d2*1
    d1 = x1.shape[0]
    d2 = x2.shape[0]
    dis = (x1*x1).dot(np.ones((1,d2))) + np.ones((d1,1)).dot((x2*x2).T) - 2*x1.dot(x2.T)
    kernel = np.exp(-1/2*dis/sigma**2)
    return kernel

def GaussianProcess(x , sigma):
    var = Kernel(x, x, sigma)
    y = np.random.multivariate_normal(np.zeros((100,)), var)
    return y

def conditional_distribution(X, x, y, sigma):
    sigma_y = Kernel(x, x, sigma)
    sigma_Y = Kernel(X, X, sigma)
    sigma_Yy = Kernel(X, x, sigma)
    sigma_yY = Kernel(x, X, sigma)
    #print(sigma_y.shape) #(5,5)
    #print(sigma_Y.shape) #(100,100)
    #print(sigma_Yy.shape) #(100, 5)
    conditional_mean = sigma_Yy.dot(np.linalg.inv(sigma_y)).dot(y)
    conditional_sigma = sigma_Y - sigma_Yy.dot(np.linalg.inv(sigma_y)).dot(sigma_yY)
    return conditional_mean, conditional_sigma

# General process
X = np.arange(-5,5,0.1)
sigma_list = [0.3, 0.5, 1.0]

for i in range(3):
    y1 = GaussianProcess(X , sigma_list[i])
    y2 = GaussianProcess(X , sigma_list[i])
    y3 = GaussianProcess(X , sigma_list[i])
    plt.plot(X,y1)
    plt.plot(X,y2)
    plt.plot(X,y3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('sigma= %f' %sigma_list[i])
    plt.show()

# Posterior distribution
x_s = np.array([-1.3, 2.4, -2.5, -3.3, 0.3])
y_s = np.array([2, 5.2, -1.5, -0.8, 0.3])
for i in range(3):
    conditional_mean, conditional_sigma = conditional_distribution(X, x_s, y_s, sigma_list[i])
    #print(conditional_mean)
    #y posterior
    y1 = np.random.multivariate_normal(conditional_mean, conditional_sigma)
    y2 = np.random.multivariate_normal(conditional_mean, conditional_sigma)
    y3 = np.random.multivariate_normal(conditional_mean, conditional_sigma)
    y4 = np.random.multivariate_normal(conditional_mean, conditional_sigma)
    y5 = np.random.multivariate_normal(conditional_mean, conditional_sigma)
    
    y_mean = np.array([y1,y2,y3,y4,y5]).mean(axis = 0)
    
    for y_ax in [y1,y2,y3,y4,y5]:
        plt.plot(X, y_ax, linewidth= 0.5)
    plt.scatter(x_s,y_s, label = 'Sample')
    plt.plot(X, y_mean, linewidth= 2, label = 'GaussianProcess_mean')
    plt.legend()
    plt.title('sigma= %f' %sigma_list[i])
    plt.show()
    