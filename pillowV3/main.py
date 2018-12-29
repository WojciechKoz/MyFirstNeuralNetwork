import numpy as np  

from mnist import MNIST

mndata = MNIST('./data')
images_full, labels_full = mndata.load_training()
images = []
labels = []

for i in range(50):
    images.append(images_full[i*100 : (i+1)*100])
    labels.append(labels_full[i*100 : (i+1)*100])

def sigmoid_prime(x):
    return np.exp(-x) / ((np.exp(-x) + 1) ** 2)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


#X = np.array([[0, 0],
#              [0, 1],
#              [1, 0],
#              [1, 1]])

#X = np.array(images)

y = []

for batch in labels:
    y.append([])
    for label in batch:
        y[-1].append([1.0 if i == label else 0.0 for i in range(10)])

y = np.array(y)

#y = np.array([[0],
#              [1],
#              [1],
#              [0]])

np.random.seed(1)

LEN = len(labels)
SIZES = [ 784, 17, 16, 10 ]

syn0 = 2 * np.random.random((SIZES[0], SIZES[1])) - 1  
syn1 = 2 * np.random.random((SIZES[1], SIZES[2])) - 1 
syn2 = 2 * np.random.random((SIZES[2], SIZES[3])) - 1  

# biases for respective layers
b0 = 2 * np.random.random((1, SIZES[1])) - 1
b1 = 2 * np.random.random((1, SIZES[2])) - 1
b2 = 2 * np.random.random((1, SIZES[3])) - 1

for i, batch in enumerate(images):
    X = np.array(batch)
    print("x:")
    print(np.shape(X))

    for j in range(500):
        l0 = X
        l1 = sigmoid(np.dot(l0, syn0) + b0)
        l2 = sigmoid(np.dot(l1, syn1) + b1)
        l3 = sigmoid(np.dot(l2, syn2) + b2)

        l3_error = (y[i] - l3)#** 2

        if j % 20 == 0:
            print(("[%d] error: " % j) + str(np.mean(np.abs(l3_error))))

        l3_delta = l3_error * sigmoid_prime(l3)
        l2_error = l3_delta.dot(syn2.T)
        l2_delta = l2_error * sigmoid_prime(l2)
        l1_error = l2_delta.dot(syn1.T)
        l1_delta = l1_error * sigmoid_prime(l1)

        syn2 += l2.T.dot(l3_delta)
        syn1 += l1.T.dot(l2_delta)
        syn0 += l0.T.dot(l1_delta)

        b0 += l1_delta.mean(axis=0)
        b1 += l2_delta.mean(axis=0)
        b2 += l3_delta.mean(axis=0)


def predict(data):
    l0 = [data]
    l1 = sigmoid(np.dot(l0, syn0) + b0)
    l2 = sigmoid(np.dot(l1, syn1) + b1)
    l3 = sigmoid(np.dot(l2, syn2) + b2)
    return np.argmax(l3)

print("Output after training: ")
print(l3)
for i, el in enumerate(l3):
    print(labels[0][i], "=", np.argmax(el), " predictions: ", el)

testing_images, testing_labels = mndata.load_testing()
correct = 0.0
for i, (image, label) in enumerate(zip(testing_images, testing_labels)):
    prediction = predict(image)
    if label == prediction:
        correct += 1.0
    print("{} = {} (correct {}%)".format(label, prediction, 100 * correct / (i + 1.0)))
