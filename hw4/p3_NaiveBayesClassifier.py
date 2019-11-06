import pandas as pd
import numpy as np
import numpy.random as r
import scipy.io as sio
import scipy.stats
import matplotlib.pyplot as plt
import random
from collections import Counter
pi = np.pi

def Quantize(x, x_median):
    for i in np.arange(x.shape[0]):
        for j in np.arange(x.shape[1]):
            if x[i, j] <= x_median[j]:
                x[i, j] = 1
            else:
                x[i, j] = 2
    return x

def CalculateLikelihood(spam_ham, obs, lld_spam_1, lld_spam_2, lld_ham_1, lld_ham_2):
    prob_product = 1
    for i in np.arange(len(obs)):
        if ((obs[i] == 1.0) & (spam_ham == 1)):
            prob_product *= lld_spam_1[i]
        elif ((obs[i] == 2.0) & (spam_ham == 1)):
            prob_product *= lld_spam_2[i]
        elif ((obs[i] == 1.0) & (spam_ham == 0)):
            prob_product *= lld_ham_1[i]
        else:
            prob_product *= lld_ham_2[i]
    return prob_product

def NaiveBayesClassifier(obs, lld_spam_1, lld_spam_2, lld_ham_1, lld_ham_2, prior):
    # Calculate the posterior probabilities of spam
    posterior_spam = CalculateLikelihood(1, obs, lld_spam_1, lld_spam_2, lld_ham_1, lld_ham_2) * prior[0]
    # Calculate the posterior probabilities of ham
    posterior_ham = CalculateLikelihood(0, obs, lld_spam_1, lld_spam_2, lld_ham_1, lld_ham_2) * prior[1] 
    if posterior_ham >= posterior_spam:
        return 0
    else:
        return 1

# load the training data 
x_train = np.load("spam_train_features.npy")
y_train = np.load("spam_train_labels.npy")

# load the test data 
x_test = np.load("spam_test_features.npy")
y_test = np.load("spam_test_labels.npy")

x_train = Quantize(x_train, np.median(x_train,axis=0))
x_test = Quantize(x_test, np.median(x_test,axis=0))

 # Split the training data. One has only spams, the other has only hams.
spam_indices = y_train == 1
x_train_spam = x_train[spam_indices,:]
x_train_ham = x_train[~spam_indices,:]
y_train_spam = y_train[spam_indices]
y_train_ham = y_train[~spam_indices]

# Get Prior Probabilities
prior_spam = Counter(y_train)[1.0] / len(y_train)
prior_ham = Counter(y_train)[0.0] / len(y_train)

lld_spam_1 = []
lld_spam_2 = []
lld_ham_1 = []
lld_ham_2 = []
# Get the likelihood of 1 and 2 for each class for each column
for j in range(x_train_spam.shape[1]):
    lld_spam_1.append(sum(x_train_spam[:,j] == 1.0) / x_train_spam.shape[0])
    lld_spam_2.append(sum(x_train_spam[:,j] == 2.0) / x_train_spam.shape[0])

for j in range(x_train_ham.shape[1]):
    lld_ham_1.append(sum(x_train_ham[:,j] == 1.0) / x_train_ham.shape[0])
    lld_ham_2.append(sum(x_train_ham[:,j] == 2.0) / x_train_ham.shape[0])

pred = []
for i in range(x_test.shape[0]):
    pred.append(NaiveBayesClassifier(x_test[i,:], lld_spam_1, lld_spam_2, lld_ham_1, lld_ham_2, [prior_spam, prior_ham]))

print("Test error: " + str(1 - sum(pred == y_test) / len(y_test)))