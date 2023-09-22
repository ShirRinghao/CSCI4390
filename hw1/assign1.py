import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

#Part 1
#Read file and format the data using numpy
file = open(sys.argv[1],'r')
epsilon = float(sys.argv[2])
data = pd.read_csv(file, delimiter = ',', skiprows = 1, usecols = range(1, 28), header = None).to_numpy()

# Compute mean value, mean matrix and center matrix
row = data.shape[0]
column = data.shape[1]
mean = np.sum(data, axis = 0) / row
variance = np.sum([np.dot(i, i) for i in data]) / row - np.dot(mean, mean)
meanMatrix = np.reshape(mean, (1, column))
centerMatrix = data - np.dot(np.ones(shape = (row, 1)), meanMatrix)

# Compute inner and outer covariance
sum = 0
for item in centerMatrix:
    sum += np.outer(item,item)
outer = sum / row
inner = np.matmul(centerMatrix.T,centerMatrix) / row

norm = np.sum(centerMatrix ** 2, axis = 0) ** 0.5
correlation = np.matmul((centerMatrix / norm).T, centerMatrix / norm)

#format to precision 4 and output for part1
np.set_printoptions(precision = 4, suppress = True)
print("Part 1(a)\nMean vector is:\n{:}\n\nTotalVarance var(D) is:\n{:.3f}\n".format(mean, variance))
print("Part 1(b)\nInner product is:\n{:}\n\nOuter product is:\n{:}\n".format(inner, outer))
print("Part 1(c)\nCorrelation matrix is:\n{:}\n".format(correlation))

# The least related data 25 and 1  --- Visibility and Appliances --- -0.0002
plt.figure(figsize=(5, 5))
plt.scatter(centerMatrix[:, 24], centerMatrix[:, 0], s = 5, c = '#1f77b4', alpha = 0.5)
plt.xlabel('Attribute 25')
plt.ylabel('Attribute 1')
plt.title("Least Correlated Attributes ([0,24])")
plt.show()

# The most related data 21 and 13  --- T1 and RH_out --- 0.9748
plt.figure(figsize=(5, 5))
plt.scatter(centerMatrix[:, 20], centerMatrix[:, 12], s = 1, c = '#1f77b4', alpha = 0.5)
plt.xlabel('Attribute 21')
plt.ylabel('Attribute 13')
plt.title("Most Correlated Attributes ([12,20])")
plt.show()

# The most anti-related data 15 and 14  ---  RH_6 and T7 --- -0.754
plt.figure(figsize=(5, 5))
plt.scatter(centerMatrix[:, 14], centerMatrix[:, 13], s = 2, c = '#1f77b4', alpha = 0.5)
plt.xlabel('Attribute 15')
plt.ylabel('Attribute 14')
plt.title("Most An-ti Correlated Attributes ([13,14])")
plt.show()

#Part2
#below is iteration and help function to solve for dominant eigenvector
def eigenvalue(A, v):
    Av = A.dot(v)
    return v.dot(Av)

# Implementation is based on online source http://mlwiki.org/index.php/Power_Iteration
# with my own modification
    v = np.ones(d) / np.sqrt(d)
    ev = eigenvalue(A, v)
    k = 0
    while True:
        k += 1
        Av = A.dot(v)
        v_new = Av / np.linalg.norm(Av)
        ev_new = eigenvalue(A, v_new)
        err = np.abs(ev - ev_new)
        #print every step to confirm it is moving in the right direction
        print("Eigen Val: {:.3f}; Error: {:.2f}; input epsilon: {:.5f};\n Eigen Vec {:};".format( ev_new,err, epsilon,v_new))
        if err < epsilon:
            break
        v = v_new
        ev = ev_new
    return v_new,v

# Calculate eigen vector, vector, scaled eigen vector and projection
# to check right or wrong
ev,v = power_iteration(inner)
ev_unit = ev / np.linalg.norm(ev)
projections = np.dot(data,ev_unit)

# Graph projection
plt.figure(figsize=(5, 5))
plt.scatter(range(len(projections)), projections,s = 10)
plt.xlabel('original data point')
plt.ylabel('scalar projection on dominant eigenvector')
plt.title("Projection graph")
plt.show()