import math
import sys
import numpy as np
import pandas as pd
from numpy import linalg as LA
import matplotlib.pyplot as plt

def kernelPCA(data, alpha, spread, flag):
    rows = data.shape[0]
    kernelMatrix = np.zeros(shape = (rows, rows))

    # Construct kernel matrix
    if(flag):
        # Kernel PCA
        # Calculate the dot product
        for i in range(rows):
            for j in range(rows):
                kernelMatrix[i,j] = np.dot(data[i],data[j])
    else:
        # Gaussian PCA
        for i in range(rows):
            for j in range(rows):
                kernelMatrix[i, j] = np.exp((-(np.linalg.norm(data[i] - data[j]) ** 2)) / (2 * spread))

    center = (np.identity(rows) - 1.0 / rows * np.ones(shape=(rows, rows)))
    centerMatrix = center @ kernelMatrix @ center
    value,vector = LA.eigh(centerMatrix)
    vector = np.fliplr(vector)
    value = value[::-1]
    # Choose eigenvectors that are greater than 0 (valid ones)
    newEigen = np.array(list(map(lambda eig: True if eig > 0 else False, value)))
    vector = vector[:, newEigen]
    value = list(filter(lambda x: x > 0, value))

    # Find variance and norm
    variance = []
    normVector = []
    for i in range(len(value)):
        variance.append(value[i] / len(value))
        normVector.append(np.multiply(math.sqrt(1 / value[i]), vector.T[i]))

    # Find the dimensions we need to capture the data
    sumValue = 0
    d_required = 0
    total_variance = sum(variance)
    for i in range(0,len(variance)):
        if (sumValue / total_variance >= alpha):
            d_required = i
            break
        sumValue = sumValue + variance[i]

    # Plot the graph
    u1 = normVector[0:2] @ centerMatrix.transpose()
    plt.figure(figsize=(6, 5))
    plt.xlabel("u1")
    plt.ylabel("u2")
    if(flag):
        plt.title("Kernel PCA")
    else:
        plt.title("Gaussian Kernel PCA")
    plt.scatter(u1[0], u1[1])
    plt.show()
    print("eigenvalues are: {:}".format(value[0:i]))
    return d_required


if __name__ == '__main__':

    # Read file and format the data using numpy
    file = open(sys.argv[1], 'r')
    alpha = float(sys.argv[2])
    spread = int(sys.argv[3])
    data = pd.read_csv(file, delimiter=',', skiprows=1, nrows = 5000, usecols=range(1, 28), header=None).to_numpy()

    # Run Kernel PCA
    d1 = kernelPCA(data, alpha, 0, 1)
    print("We need {:} dimensions to capture alpha = {:.3f} fraction of total variance using kernel PCA.".format(d1, alpha))

    # Run Gaussian Kernel PCA
    d2 = kernelPCA(data, alpha, spread, 0)
    print("We need {:} dimensions, alpha = {:.3f}, and spread value {:} to find the best result using Gaussian PCA.".format(d2, alpha, spread))


