import sys
import math
import numpy as np
import pandas as pd
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random as ran

# Function that generates random data for train and test set
def random(data):
    np.random.shuffle(data)
    train = []
    test = []
    for i in range(0, int(len(data) * 0.7)):
        train.append(data[i])
    for i in range(int(len(data) * 0.7 + 1), len(data)):
        test.append(data[i])
    return train, test

# Determine the sigmoid value
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def generate(x, y, w):
    # return the predit 0 or 1
    augX = aug(x)
    return sigmoid(np.matmul(augX, w)) >= 0.5

# Try to find a fit for each step according to the given input values
def computeValue(x, y, eta, epsilon, maxiter):
    augX = np.array(aug(x))
    weights = np.zeros(augX.shape[1])  # initial weight vector

    i = 0
    # For randomly going over the data
    r = list(zip(augX, y))
    ran.shuffle(r)
    augX, y = zip(*r)
    n = x.shape[0]
    while(1):
        i += 1
        if(i > maxiter):
            break
        prevNorm = np.linalg.norm(weights)
        for j in range(n):
            newX, newY = augX[j], y[j]
            tmp = sigmoid(np.dot(weights.T, newX))
            gradient = (newY - tmp) * newX
            weights += eta * gradient
        if epsilon > abs(np.linalg.norm(weights) - prevNorm):
            print("Converge at this point!")
            break
    return weights


def aug(x):
    n, d = x.shape
    augX = np.ones((n, d + 1))
    augX[:, 1:] = x  # augment the data
    return np.array(augX)

if __name__ == '__main__':
    file = open(sys.argv[1], 'r')
    data = pd.read_csv(file, delimiter=',', skiprows=1, usecols=range(1, 28), header=None).to_numpy()

    # set data value to either 0 or 1 according to the given instruction
    for row in data:
        if (row[0] <= 50):
            row[0] = 1
        else:
            row[0] = 0

    # Select the data randomly from the entire dataset
    train, test = random(data)
    xtrain = np.array(train)[:, 1:]
    ytrain = np.array(train)[:, 0]
    xtest = np.array(test)[:, 1:]
    ytest = np.array(test)[:, 0]

    # Test result for different input values on training and testing
    eta = float(sys.argv[2])
    eps = float(sys.argv[3])
    maxiter = int(sys.argv[4])
    w = computeValue(xtrain, ytrain, eta, eps, maxiter)
    print("Weight is: ", w)
    print("eta: {:}, eps: {:}, maxiter: {:}".format(eta, eps, maxiter))
    trainPred = list(generate(xtrain, ytrain, w) - ytrain)
    trainAccur = trainPred.count(0) / len(trainPred)
    testPred = list(generate(xtest, ytest, w) - ytest)
    testAccur = testPred.count(0) / len(testPred)

    print("Traning data has an accuracy of {:} ".format(trainAccur))
    print("Testing data has an accuracy of {:} ".format(testAccur))
