import numpy as np 
import matplotlib.pyplot as plt
np.random.seed(0) 
from scipy.stats import norm 

def LogLikelihood(n,K,ee,y,we,be,ve):
    ll = 0
    for i in range(0,n):
        tmp = 0
        for k in range(0,K):
            tmp += ee[k]*norm.pdf(y[i],we[k]*x[i] + be[k],np.sqrt(ve[k])) 
        ll = ll + np.log(tmp)
    return ll

#E step
def Gam(n,K,ee,y,we,be,ve):
    gam = np.zeros([n,K])
    for i in range(0,n):
        denom = 0.0
        for l in range(0,K):
            denom += ee[l]*norm.pdf(y[i],we[l]*x[i] + be[l],np.sqrt(ve[l])) 
        for k in range(0,K):
            gam[i,k] = ee[k]*norm.pdf(y[i],we[k]*x[i]+ be[k],np.sqrt(ve[k]))/denom
    print gam.shape
    return gam

#M step
#1.update e
def UpdateE(gam):
    ee = np.mean(gam,axis=0)
    return ee

#2.update w,b,var
def UpdateWBVar(n,K,gam,y,we,be,ve):
    A = np.concatenate((x[:,np.newaxis], np.ones([n,1])),axis=1)
    for k in range(0,K):
        C = np.diag(gam[:,k])
        yT = y[:,np.newaxis]
        z = np.linalg.inv(A.T.dot(C).dot(A)).dot(A.T).dot(C).dot(yT)
        #Weighted least squares
        we[k] = z[0]
        be[k] = z[1]
        ve[k] = ((yT - A.dot(z)).T.dot(C).dot(yT - A.dot(z)))/np.sum(gam[:,k])
    return we, be, ve

n = 200 #sample size 
K = 2 #number of lines 
e = np.array([0.7,0.3]) #mixing weights 
w = np.array([-2,1]) #slopes of lines 
b = np.array([0.5,-0.5]) #offsets of lines 
v = np.array([0.2,0.1]) #variances 
x = np.zeros([n]) 
y = np.zeros([n]) 

for i in range(0,n):
    x[i] = np.random.rand(1) 
    if np.random.rand(1) < e[0]:
        y[i] = w[0]*x[i] + b[0] + np.random.randn(1)*np.sqrt(v[0])
    else:
        y[i] = w[1]*x[i] + b[1] + np.random.randn(1)*np.sqrt(v[1])
plt.plot(x,y,'bo') 
t = np.linspace(0, 1, num=100) 
plt.plot(t,w[0]*t+b[0],'k') 
plt.plot(t,w[1]*t+b[1],'k')
plt.show()

#EM algorithm
it = 500 #max number of iterations
ee = np.array([0.5,0.5])
we = np.array([1.0,-1.0])
be = np.array([0.0,0.0])
ve = np.array([np.var(y),np.var(y)]) 
ll = np.zeros([it])

for t in range(0,it):
    print "number of iterations: " + str(t) 
    ll[t] = LogLikelihood(n,K,ee,y,we,be,ve)
    print(ll[t])
    if t>=1:
        if (ll[t]-ll[t-1]) < 10**(-4):
            break
    gam = Gam(n,K,ee,y,we,be,ve)
    ee = UpdateE(gam)
    #print(ee.shape)
    we,be,ve = UpdateWBVar(n,K,gam,y,we,be,ve)
    #print(we.shape)
    #print(be.shape)
    #print(we.shape)

print e,w,b,v
print ee,we,be,ve

plt.clf()
plt.plot(ll[0:t])
plt.show()

plt.clf()
plt.plot(x,y,'bo')
t = np.linspace(0, 1, num=100) 
plt.plot(t,w[0]*t+b[0],'k')
plt.plot(t,w[1]*t+b[1],'k')
plt.plot(t,we[0]*t+be[0],'r:')
plt.plot(t,we[1]*t+be[1],'r:')
plt.show()