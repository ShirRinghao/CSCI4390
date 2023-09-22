import random
import numpy as np
import matplotlib.pyplot as plt
import statistics as stats

# Calculate angles and frequency
def calculation(d):
    # Create a list for angle, dictionary for frequency and get random number list
    list = []
    dict = {}
    pairs = randomGenerator(d)
    # Calculator angles between each pair using for loop
    for i in range(len(pairs)):
        n1 = np.linalg.norm(pairs[i][0])
        n2 = np.linalg.norm(pairs[i][1])
        divisor = n1 * n2
        dotproduct= np.dot(pairs[i][0],pairs[i][1])
        r = np.arccos(dotproduct/divisor)
        list.append(np.rad2deg(r))

    # Count the frequency
    for item in list:
        if item in dict.keys():
            dict[item] = dict[item] + 1
        else:
            dict[item] = 1
    return dict, list, len(list)

# generate 100000 random number and store them into a list
def randomGenerator(d):
    list = []
    for i in range(100000):
        tmp1 = []
        tmp2 = []
        # for every dimension, generate a random number accordingly
        for j in range(d):
            tmp1.append(random.choice([1, -1]))
            tmp2.append( random.choice([1, -1]))
        list.append([tmp1, tmp2])
    return list

if __name__ == '__main__':
    # Part 2 produce graphs for 3 different values of d: 10, 100, 1000
    dlist = [10, 100, 1000]
    for i in range(len(dlist)):
        dict, list, length = calculation(dlist[i])
        keys = dict.keys()
        for item in keys:
            dict[item] = dict[item] / length

        # Output
        print("Minimum value is: {:}\nMaximum value is: {:}".format(min(list), max(list)))
        print("Value range is: {:}\nMean is: {:}, Variance is: {:}".format(abs(min(list) - max(list)), stats.mean(list), stats.variance(list)))

        # Plot the graph
        plt.figure()
        plt.xlabel("Angle in degree")
        plt.ylabel("Probability")
        plt.bar(keys, dict.values(), width = 10)
        plt.title("PMF of Dimensions {:}".format(dlist[i]))
        plt.show()
        dict.clear()