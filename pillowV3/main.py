import numpy as np  


def sigmoid_prime(x):
    return np.exp(-x) / ((np.exp(-x) + 1) ** 2)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

np.random.seed(1)

syn0 = 2 * np.random.random((2, 4)) - 1  
syn1 = 2 * np.random.random((4, 1)) - 1 

b0 = 2 * np.random.random((1, 4)) - 1
b1 = 2 * np.random.random((1, 1)) - 1

b0 = np.vstack([b0] + [b0[0]]*3)
b1 = np.vstack([b1] + [b1[0]]*3)

for j in range(500):
    l0 = X
    l1 = sigmoid(np.dot(l0, syn0) + b0)
    l2 = sigmoid(np.dot(l1, syn1) + b1)

    l2_error = (y - l2)#** 2

    if (j % 10000) == 0:
        print("error: " + str(np.mean(np.abs(l2_error))))

    l2_delta = l2_error * sigmoid_prime(l2)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * sigmoid_prime(l1)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

    b0 += l1_delta
    b1 += l2_delta

print("Output after training: ")
print(l2)
