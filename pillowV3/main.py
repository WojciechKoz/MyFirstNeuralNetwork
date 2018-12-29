import numpy as np  

from mnist import MNIST

mndata = MNIST('./data')
images_full, labels_full = mndata.load_training()

images = images_full[:100]
labels = labels_full[:100]

def sigmoid_prime(x):
    return np.exp(-x) / ((np.exp(-x) + 1) ** 2)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


#X = np.array([[0, 0],
#              [0, 1],
#              [1, 0],
#              [1, 1]])

X = np.array(images)

y = []
for label in labels:
    y.append([1.0 if i == label else 0.0 for i in range(10)])

y = np.array(y)

#y = np.array([[0],
#              [1],
#              [1],
#              [0]])

np.random.seed(1)

LEN = len(labels)
SIZES = [ 784, 16, 10 ]

syn0 = 2 * np.random.random((SIZES[0], SIZES[1])) - 1  
syn1 = 2 * np.random.random((SIZES[1], SIZES[2])) - 1 

# biases for respective layers
b0 = 2 * np.random.random((1, SIZES[1])) - 1
b1 = 2 * np.random.random((1, SIZES[2])) - 1

# extend the biases 
b0 = np.vstack([b0] + [b0[0]] * (LEN - 1))
b1 = np.vstack([b1] + [b1[0]] * (LEN - 1))

def average_rows(b):
    height = np.shape(b)[0]
    b = b.mean(axis=0)
    #print(b)
    return np.array([b] * height)

for j in range(500):
    l0 = X
    l1 = sigmoid(np.dot(l0, syn0) + b0)
    l2 = sigmoid(np.dot(l1, syn1) + b1)

    l2_error = (y - l2)#** 2

    print(("[%d] error: " % j) + str(np.mean(np.abs(l2_error))))

    l2_delta = l2_error * sigmoid_prime(l2)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * sigmoid_prime(l1)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

    b0 += l1_delta
    b0 = average_rows(b0)
    b1 += l2_delta
    b1 = average_rows(b1)


def predict(data):
    l0 = [data]
    l1 = sigmoid(np.dot(l0, syn0) + b0[0])
    l2 = sigmoid(np.dot(l1, syn1) + b1[0])
    return np.argmax(l2)

print("Output after training: ")
print(l2)
for i, el in enumerate(l2):
    print(labels[i], "=", np.argmax(el), " predictions: ", el)

for i in range(101, 200):
    print(labels_full[i], predict(images_full[i]))
