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

def Kernel(x, sigma):
    d  = x.shape[0]
    kernel =  np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            kernel[i][j] = np.exp(-1/(2*sigma**2) * (x[j] - x[i])**2)
    return kernel

def GaussianProcess(x , sigma):
    var = Kernel(x, sigma)
    y = np.random.multivariate_normal(np.zeros((100,)), var)
    return y

def multivariate_gaussian(x, mean, sigma):
    d = sigma.shape[0]
    quadratic = (x-mean).T.dot(np.linalg.inv(sigma)).dot(x-mean)
    return 1/(2*pi)**(d/2)*1/np.sqrt(np.linalg.det(sigma)) * np.exp(-1/2*quadratic)

def marginal_distribution(mean, sigma, indice):
    index = np.tile(0,mean.shape[0])
    for i in indice:
        index[i] = 1 
    index_sig = index.reshape((1,-1))
    marginal_mean = np.ma.array(mean, mask= np.subtract(1, index)).compressed()
    marginal_sigma = np.ma.array(sigma, mask = np.subtract(1,index_sig.T.dot(index_sig))).compressed().reshape((sum(index),sum(index)))
    return marginal_mean, marginal_sigma

def covaraince(sigma, indice_U, indice_V):
    index_matrix = np.zeros((sigma.shape[0],sigma.shape[0]))
    index_matrix[:,indice_U]=1
    index_matrix[indice_V,:]=1
    UV_sigma = np.ma.array(sigma, mask = index_matrix).compressed().reshape((len(indice_U),len(indice_V)))
    return UV_sigma

def conditional_distribution(mean, sigma, indice_U, indice_V, V):
    V_mean, V_sigma = marginal_distribution(mean, sigma, indice_V)
    U_mean, U_sigma = marginal_distribution(mean, sigma, indice_U)
    UV_sigma = covaraince(sigma, indice_U, indice_V)
    U_conditional_mean = U_mean + UV_sigma.dot(np.linalg.inv(V_sigma)).dot(V-V_mean)
    U_conditional_sigma = U_sigma - UV_sigma.dot(np.linalg.inv(V_sigma)).dot(UV_sigma.T)
    return U_conditional_mean, U_conditional_sigma

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
indice_V = np.array([37,74,25,17,53])
indice_U = np.delete(np.arange(0,100,1) , indice_V)
for i in range(3):
    var = Kernel(X, sigma_list[i])
    #print(var.shape)
    conditional_mean, conditional_sigma = conditional_distribution(np.zeros((100,1)), var, list(indice_U), list(indice_V), y_s)
    #print(conditional_mean.shape, conditional_sigma.shape)
    
    #y posterior
    new_x = X[[i for i in range(len(X)) if i not in indice_V]]
    y1 = np.random.multivariate_normal(conditional_mean, conditional_sigma)
    y2 = np.random.multivariate_normal(conditional_mean, conditional_sigma)
    y3 = np.random.multivariate_normal(conditional_mean, conditional_sigma)
    y4 = np.random.multivariate_normal(conditional_mean, conditional_sigma)
    y5 = np.random.multivariate_normal(conditional_mean, conditional_sigma)
    
    y_mean = np.array([y1,y2,y3,y4,y5]).mean(axis = 0)
    
    #sort x and y
    X_all = np.append(new_x,X[indice_V])
    arg = np.argsort(X_all)
    
    for y_ax in [y1,y2,y3,y4,y5]:
        p_y = np.append(y_ax,  y_s)
        plt.plot(X_all[arg], p_y[arg], linewidth= 0.5)
    plt.scatter(x_s,y_s, label = 'Sample')
    p_y = np.append(y_mean,  y_s)
    plt.plot(X_all[arg], p_y[arg], linewidth= 2, label = 'GP_mean')
    plt.legend()
    plt.title('sigma= %f' %sigma_list[i])
    plt.show()
    
