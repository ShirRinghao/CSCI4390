import sys
import numpy as np
import pandas as pd

# Randomly select 70% data for training, and 30% for testing
def random(data):
    np.random.shuffle(data)
    train = []
    test = []
    for i in range(0, int(len(data) * 0.7)):
        train.append(data[i])
    for i in range(int(len(data) * 0.7 + 1), len(data)):
        test.append(data[i])
    return train, test

# Solve, using backsolve to calculate weight and norm weight
def solve(x, y):
    # Add a biased column
    x_aug = np.insert(x, 0, 1, axis=1)
    q, r = np.linalg.qr(x_aug)
    rhs = np.multiply(1 / np.diag(np.matmul(q.T, q)), np.matmul(q.T, y))
    weight = np.ones(r.shape[1])
    # Back solve
    for i in range(len(weight) - 1, -1, -1):
        weight[i] = (rhs[i] - np.dot(r[i, :], weight) + r[i, i]) / r[i, i]
    normWeight = np.linalg.norm(weight)
    return weight, normWeight

# predict the output on testing data set
def predict(x, y, w):
    # Add a biased column
    x_aug = np.insert(x, 0, 1, axis=1)
    sse = np.linalg.norm(np.matmul(x_aug, w) - y) ** 2
    tss = np.linalg.norm(y - np.mean(y)) ** 2
    mse = tss / x.shape[0]
    rSquare = (tss - sse) / tss
    return sse, mse, rSquare

if __name__ == '__main__':
    file = open(sys.argv[1], 'r')
    data = pd.read_csv(file, delimiter=',', skiprows=1, usecols=range(1, 28), header=None).to_numpy()

    # Randomly select data, and take out x and y values
    train, test = random(data)
    train_x = np.array(train)[:, 1:len(data)]
    train_y = np.array(train)[:, 0]
    test_x = np.array(test)[:, 1:len(data)]
    test_y = np.array(test)[:, 0]

    # Output weight and norm_weight
    weight, normWeight = solve(train_x, train_y)
    print("Training data: weight vector is {:}\n\nTraining data: L2 norm weight is {:.5f}\n".format(weight, normWeight))

    # Output for train and test result
    train_sse, train_mse, train_square = predict(train_x, train_y, weight)
    test_sse, test_mse, test_square = predict(test_x, test_y, weight)
    print("Training result: SSE is {:.5f}, MSE is {:.5f}, R Square is {:.5f}\n".format(train_sse, train_mse, train_square))
    print("Testing result: SSE is {:.5f}, MSE is {:.5f}, R Square is {:.5f}\n".format(test_sse, test_mse, test_square))