import sys
import numpy as np
import pandas as pd
import random as random

# initailize the bias vectors
def initialize(m,r,c):
    Bh = []
    for i in range(0, m):
        Bh.append(random.random())
    Bo = []
    for i in range(0, r):
        Bo.append(random.random())
    Wh = np.random.randn(c - 1, m)
    Wo = np.random.randn(m, r)
    return Bh, Bo, Wh, Wo

def filter_fun(SigO):
    for i in range(0,len(SigO)):
        if(SigO[i] < 0.000001):
            SigO[i] = 0
        else: 
            SigO[i] = 1
    return SigO

# Get data for x and y
def fetch(path):
    y = data[:,0]
    x = data[:,1:]
    return x,y

def RELU(l):
    r = []
    for x in l:
        if x > 0:
            r.append(x)
        else:
            r.append(0)
    return r

def Partial_RELU(l):
    r = []
    for x in l:
        if x > 0:
            r.append(1)
        else:
            r.append(0)
    return r

def aug(x):
    n, d = x.shape
    augX = np.ones((n, d + 1))
    augX[:, 1:] = x
    return np.array(augX)
    
def random_sort(data):
    # Sort the input data into 70% train, 30% test
    np.random.shuffle(data)
    train = []
    test = []
    for i in range(0, int(len(data) * 0.7)):
        train.append(data[i])
    for i in range(int(len(data) * 0.7 + 1), len(data)):
        test.append(data[i])
    return train, test

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    

def MLP(x, y, m, step, maxiter):
    # initialize the data and compute the values
    r, c = data.shape
    Bh, Bo, Wh, Wo = initialize(m,r,c);
    for i in range(maxiter):
        # permutation in random order (stochastic)
        p = np.random.permutation(len(y))
        x = x[p]
        y = y[p]
        for j in range(len(y)):
            tmp = []
            # Net hidden layer
            SigY = y[j]
            SigZ = RELU(Bh + np.matmul(Wh.T,x[j]))
            # Net output layer
            SigO = filter_fun(np.concatenate(([], sigmoid(Bo + np.matmul(Wo.T,SigZ))), axis = None))
            SigY = np.full((1, r), y[j])
            deltaO = SigO - SigY
            tmp = np.concatenate(([], np.matmul(Wo,deltaO.T)), axis = None)
            deltaH = np.multiply(Partial_RELU(Bh + np.matmul(Wh.T, x[j])), tmp)
            # Gradient descent
            Bo = Bo - step * deltaO
            Bh = Bh - step * deltaH
            tmp = []
            tmp.append(SigZ)
            # Gradient descent for weight matrices
            Wo = Wo - step * np.matmul(np.transpose(tmp), deltaO)
            Wh = Wh - step * np.outer(x[j], deltaH)
    return Wh, Wo, Bh, Bo

if __name__ == '__main__':
    # Input
    file = open(sys.argv[1], 'r')
    ETA = float(sys.argv[2])
    MAXITER = int(sys.argv[3])
    HIDDENSIZE = int(sys.argv[4])
    data = pd.read_csv(file, delimiter=',', skiprows=1, usecols=range(1, 28), header=None).to_numpy()

    # Set data value to 0 or 1 depends on its value
    for row in data:
        if (row[0] <= 50):
            row[0] = 1
        else:
            row[0] = 0

    train, test = random_sort(data)
    trainX = np.array(train)[:, 1:]
    trainY = np.array(train)[:, 0]
    testX = np.array(test)[:, 1:]
    testY = np.array(test)[:, 0]

# Run SGD algorithm
Wh, Wo, Bh, Bo = MLP(trainX,trainY, HIDDENSIZE, ETA, MAXITER)

z = RELU(sum(np.transpose(np.matmul(Wh.T,trainX.T))) + Bh) 
o = sigmoid(sum(np.transpose(np.multiply(z,Wo.T))) + Bo)
arr = []
predict = []
o = np.concatenate((arr, o), axis = None)
for i in o:
    if(i >= 0.5):
        a = 1
    else:
        a = 0
    predict.append(a)
    
# Output the result
print("Hiddensize: {}\nETA: {}\nMAITER: {}\n".format(HIDDENSIZE, ETA, MAXITER))
print("Wh : {:}\nBh {:}\nWo {:}\nBo {:}".format(Wh, Bh, Wo, Bo))
print("The final accuracy is ", list(predict - data[:,0]).count(0)/len(predict))