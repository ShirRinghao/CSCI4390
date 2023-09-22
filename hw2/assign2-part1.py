# Load package
import sys
import numpy as np
import pandas as pd
from numpy import linalg as LA
import matplotlib.pyplot as plt

# written according to pseudocode from the given textbook
def part1_pca(data, alpha):
    # Calculate center matrix, covariance, variance, and eigenvector
    centerMatrix = data - np.mean(data, 0)
    covariance = np.cov(np.transpose(centerMatrix))
    variance = np.trace(covariance)
    value, eigenvector = LA.eig(covariance)

    # Loop through to find the minimum dimension required
    sumValue = 0
    d_required = 0
    for i in range(0,len(value) + 1):
        if (alpha <= (sumValue / variance)):
            d_required = i
            break
        sumValue = sumValue + value[i]

    # Calculate MSE and matrix of eigenvector
    mse = variance - sum(value[0:3])
    u1 = eigenvector[0:2] @ centerMatrix.transpose()
    u2 = eigenvector[:,0] @ centerMatrix.transpose()
    u3 = eigenvector[:,1] @ centerMatrix.transpose()
    # Plot the graph and output
    plt.figure(figsize=(6, 5))
    plt.xlabel("u1")
    plt.ylabel("u2")
    plt.title("Standard PCA (two columns of eigenvector)")
    plt.scatter(u2,u3)
    plt.show()
    print("We need {:} dimensions to capture alpha = {:.3f} fraction of total variance.\nMean Squared Error is {:5f}.".format(d_required, alpha, mse))
    return u1


if __name__ == '__main__':
    # Read file and format the data using numpy
    file = open(sys.argv[1], 'r')
    alpha = float(sys.argv[2])
    data = pd.read_csv(file, delimiter=',', skiprows=1, usecols=range(1, 28), header=None).to_numpy()

    # Part 1, PCA function
    u = part1_pca(data, 0.975)

    # Plot the graph
    plt.figure(figsize=(6, 5))
    plt.xlabel("u1")
    plt.ylabel("u2")
    plt.title("Projection of data")
    plt.scatter(u[0],u[1])
    plt.show()