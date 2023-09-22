import sys
import math
import numpy as np
import pandas as pd
from numpy import linalg as LA
import matplotlib.pyplot as plt
import time
import random 

def classfy(data):
    # shuflle the data into train and test
    np.random.shuffle(data)
    train = []
    test = []
    for i in range(0, int(len(data) * 0.7)):
        train.append(data[i])
    for i in range(int(len(data) * 0.7 + 1), len(data)):
        test.append(data[i])
    return train, test

def aug(x):
    n, d = x.shape
    augX = np.ones((n, d + 1))
    augX[:, 1:] = x  # augment the data
    return np.array(augX)

def KERNEL(flag,loss,data_x,spread,C):
    # calculate the different kernel matrix
    rows = data_x.shape[0]
    data_x = aug(data_x)
    kernelMatrix = np.zeros(shape = (rows, rows))
    if(flag == 'linear'):
        # Kernel PCA
        # Calculate the dot product
        for i in range(rows):
            for j in range(rows):
                kernelMatrix[i,j] = np.dot(data_x[i],data_x[j])
                if loss == 'quadratic':
                    if i == j:
                        delta = 1
                    else:
                        delta = 0
                    kernelMatrix[i,j] = kernelMatrix[i,j] + (1/2*C)*delta
    else:
        # Gaussian PCA
        for i in range(rows):
            for j in range(rows):
                kernelMatrix[i, j] = np.exp((-(np.linalg.norm(data_x[i] - data_x[j]) ** 2)) / (2 * spread))
                if loss == 'quadratic':
                    if i == j:
                        delta = 1
                    else:
                        delta = 0
                    kernelMatrix[i,j] = kernelMatrix[i,j] + (1/2*C)*delta #constrain
    return kernelMatrix

def SVM(trainX, trainY, eps, constrain, maxiter, loss, spread,flag):
    kernel_matrix = KERNEL(flag,loss,trainX,spread,constrain)
    trainX = aug(trainX)
    r,c = np.shape(trainX)
    size = 1/ np.diag(kernel_matrix) #f or the step size
    alpha = np.concatenate(([], np.zeros((r,1))), axis=None)
    for t in range(0,maxiter):
        perm = np.random.permutation(r)
        trainX = trainX[perm] #random doing
        trainY = trainY[perm]
        size = size[perm]
        alpha_old = np.copy(alpha)
        for k in range(0, r):
            total = 0
            for i in range(0,r):
                total  = total + (alpha[i]*trainY[i]*kernel_matrix[i,k])
            alpha[k] = alpha[k] + size[k]* (1-total*trainY[k])
            if loss == 'hinge': # constrain
                if alpha[k]< 0:
                    alpha[k] = 0
                if alpha[k] > constrain:
                    alpha[k] = constrain                   
            if loss == 'quadratic':
                if alpha[k] > constrain:
                    alpha[k] = constrain   
        if np.linalg.norm(alpha- alpha_old)< eps: #stopping
            break;
            
    predict(alpha, trainX, trainY, kernel_matrix )
    if flag == 'linear': #print the linear weight and bias
        w = np.zeros((26,1))
        w = np.concatenate(([], w), axis=None)
        for i in range(0,r):
            w = w + np.multiply(alpha[i]* trainY[i],alpha[i]* trainY[i][1:])
        b = []
        for i in range(0,r):
            b.append(trainY[i]-np.dot(w,trainX[i][1:]))
        b_avg =np.mean(b)
        print("The weight vector is: ",w)
        print('The bias vector is: ', b)
        print("The bias average is: ", b_avg)
    
    if loss == 'hinge': # compute the hinge SVM number
        count = list(filter(lambda x: constrain > x > 0, alpha))
        print("The number of SVM for hinge is ", len(count))
        
    if loss == 'quadratic':# compute the quadratic SVM number
        count = list(filter(lambda x: x > 0, alpha))
        print("The number of SVM for quadratic is ", len(count))

def predict(alpha, trainX, trainY, kernel_matrix):
    # predict the correct label
    r,c = np.shape(trainX)
    predict = [] 
    for i in range(0,r):
        sign = 0
        for j in range(0,r):
            if alpha[j] >0:
                sign = sign + alpha[j]*trainY[j]*kernel_matrix[j,i]
        if(sign < 0):
            sign = -1
        else:
            sign = 1
        predict.append(sign)
    print("The final accuracy : ", list(predict-trainY).count(0)/len(trainY))

if __name__ == '__main__':
    file = sys.argv[1]
    loss = sys.argv[2]
    print("loss is -------- ", loss)
    c = float(sys.argv[3])
    print("c is -------- ", c)
    eps = float(sys.argv[4])
    print("eps is -------- ", eps)
    MAXITER = int(sys.argv[5])
    print("maxiter is -------- ", MAXITER)
    Kernel = sys.argv[6]
    print("kernel is -------- ", Kernel)
    KERNEL_PARAM = int(sys.argv[7])
    print("spread is -------- ", KERNEL_PARAM)
    
    data = pd.read_csv(file, delimiter=',', skiprows=1,nrows = 5000, usecols=range(1, 28), header=None).to_numpy()

    # set data value to either 0 or 1 according to the given instruction
    for row in data:
        if (row[0] <= 50):
            row[0] = 1
        else:
            row[0] = -1

    # Select the data randomly from the entire dataset
    train, test = classfy(data)
    trainX = np.array(train)[:, 1:]
    trainY = np.array(train)[:, 0]
    testX = np.array(test)[:, 1:]
    testY = np.array(test)[:, 0]
    print("Training")
    print("-----------------------------------------------------")
    SVM(trainX, trainY, eps, c, MAXITER,loss, KERNEL_PARAM,Kernel)
    print("Testing")
    print("-----------------------------------------------------")
    SVM(testX, testY, eps, c, MAXITER,loss, KERNEL_PARAM,Kernel)
    
    