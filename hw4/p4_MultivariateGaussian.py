import pandas as pd
import numpy as np
import numpy.random as r
import scipy.io as sio
import scipy.stats
import matplotlib.pyplot as plt
import random
from collections import Counter
pi = np.pi

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

# marginal distribution
mean = np.array([0,0]).reshape((2,1))
sigma = np.array([1,0.5,0.5,1]).reshape((2,2))
indice = [0]
marginal_mean, marginal_sigma = marginal_distribution(mean, sigma, indice)
X = np.arange(-2,2,0.1)
pdf = []
for x in X:
    pdf.append(multivariate_gaussian(x, marginal_mean, marginal_sigma))

plt.plot(X,pdf)
plt.xlabel('x1')
plt.ylabel('p(x)')
plt.show()

# conditional distribution
mean = np.array([0.5,0,-0.5,0]).reshape((4,1))
sigma = np.array([1,0.5,0,0,0.5,1,0,1.5,0,0,2.0,0,0,1.5,0,4]).reshape((4,4))
V = np.array([0.1,-0.2])
indice_U = [0,3]
indice_V = [1,2]
U_conditional_mean, U_conditional_sigma = conditional_distribution(mean, sigma, indice_U, indice_V, V)

x1 = np.arange(-2,2,0.1)
x4 = np.arange(-2,2,0.1)
con_pdf = np.zeros((len(x1),len(x4)))
for i in range(len(x1)):
    for j in range(len(x4)):
        x = np.array([x1[i],x4[j]]).reshape((2,1))
        u_pdf = multivariate_gaussian(x, U_conditional_mean.reshape((2,1)), U_conditional_sigma.reshape((2,2))).reshape(-1)
        con_pdf[j][i] = u_pdf[0]

plt.contour(x1,x4,con_pdf)
plt.xlabel('x1')
plt.ylabel('x4')
plt.show()'x3')
plt.show()