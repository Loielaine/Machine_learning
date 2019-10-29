import pandas as pd
import numpy as np
import scipy.io
import matplotlib.pyplot as plt


def ExpTerm(t):
    return np.exp(-t)

def Gradient(X, y, theta, c):
    p,n = X.shape
    ythetax = theta.transpose().dot(X) * y #n*n 
    g1= X.dot((ExpTerm(ythetax)/(1+ ExpTerm(ythetax))*(-y)).transpose())
    #print(g1.shape)
    g2 = 2 * c * theta
    g = g1 + g2
    return g #(p+1)*1
    
def Hessian(X, y, theta, c):
    p,n = X.shape
    ythetax = (theta.T.dot(X))*y #n*n 
    sig = (ExpTerm(ythetax)/(1+ExpTerm(ythetax))**2)*X
    h1 = sig.dot(X.transpose())
    h2 = 2 * c * np.identity(p)
    h = h1 + h2
    return h.reshape((p,p)) #(p+1)*(p+1)

    
def SolveRegularizedNewtonMethod(X, y, eta, eps, c, theta0, max_iterarion):
    p,n = X.shape 
    X0 = np.ones((1,n))
    Xnew = np.vstack((X0,X))
    i = 0
    for i in range(max_iterarion):
        g = Gradient(Xnew, y, theta0, c) #(p+1)*1
        #print(g.shape)
        h = Hessian(Xnew, y, theta0, c)
        hinv =  np.linalg.inv(h) #(p+1)*(p+1)
        #print(hinv.shape)
        theta = theta0 - eta * np.dot(hinv, g)
        i = i+1
        print('Iteration: %d' %i)
        change = np.linalg.norm(theta - theta0)
        print(change)
        if change < eps:
            theta0 = theta
            break;
        else: 
            theta0 = theta  
    return theta

def CalculateAccuracy(X, y, theta):
    p,n = X.shape 
    X0 = np.ones((1,n))
    Xnew = np.vstack((X0,X))
    xtheta = theta.T.dot(Xnew) #n*1
    y_predicted = 1/(1+ExpTerm(xtheta)) 
    y_predicted = y_predicted >0.5 
    #print(y_predicted)
    accuracy = np.mean(y_predicted == y)
    print("The error rate is %f" %(1-accuracy))
    return accuracy

def CalculateConfidence(X, y, theta):
    p,n = X.shape 
    X0 = np.ones((1,n))
    Xnew = np.vstack((X0,X))
    xtheta = theta.T.dot(Xnew) #n*1
    y_predicted = 1/(1+ExpTerm(xtheta)) 
    y_predicted = y_predicted >0.5 
    y_test = y==1
    x_incorrect = X[:,(y_predicted != y_test).reshape(n,)]
    y_incorrect = y[y_predicted != y_test].reshape(1,-1)
    x_incorrect_new = np.vstack((np.ones(x_incorrect.shape[1]),x_incorrect))
    x_incorrect_theta = theta.T.dot(x_incorrect_new) 
    prob = 1/(1+ExpTerm(x_incorrect_theta)) 
    confidence = np.abs(prob-0.5)
    #print(confidence)
    prob_sorted = np.take_along_axis(prob,np.argsort(confidence,axis = 1),axis =1)
    print(prob_sorted)
    x_incorrect_sorted = np.take_along_axis(x_incorrect,np.argsort(confidence,axis = 1),axis =1)
    y_incorrect_sorted = np.take_along_axis(y_incorrect,np.argsort(confidence,axis = 1),axis =1)
    return x_incorrect_sorted, y_incorrect_sorted


mnist_49_3000 = scipy.io.loadmat('mnist_49_3000.mat')
x = mnist_49_3000['x']
y = mnist_49_3000['y']
#y[y<0] = 0
p,n = x.shape


train_x = x[:,0:2000]
test_x = x[:,2000:]
train_y = y[:,0:2000]
test_y = y[:,2000:]


i = 0
plt.imshow(np.reshape(x[:,i], (int(np.sqrt(p)), int(np.sqrt(p)))))
plt.show()


eta = 1
eps = 10**(-6)
c = 10
max_iterarion = 50 
theta0 = np.zeros((x.shape[0]+1,1))
theta = SolveRegularizedNewtonMethod(train_x, train_y, eta, eps, c, theta0, max_iterarion)


accuracy = CalculateAccuracy(test_x, test_y==1 , theta)


x_incorrect_sorted, y_incorrect_sorted = CalculateConfidence(test_x, test_y, theta)


fig, axes = plt.subplots(4,5)
k = -1
for i in range(0,4):
    for j in range(0,5):
        axes[i,j].imshow(np.reshape(x_incorrect_sorted[:,k],(int(np.sqrt(p)),int(np.sqrt(p)))))
        axes[i,j].set_title("9" if y_incorrect_sorted[:,k] == 1 else "4")
        k = k-1


