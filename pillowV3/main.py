#!/usr/bin/env python3
# -*- coding: utf8 -*-
from __future__ import print_function # new print() on python2
from datetime import datetime
import sys
import numpy as np  

from mnist import MNIST

# Display full arrays
np.set_printoptions(threshold=np.inf)

mndata = MNIST('./data')
images_full, labels_full = mndata.load_training()
images = []
labels = []

# dynamic arguments
batch_size = int(sys.argv[1])
size_1 = int(sys.argv[2])
size_2 = int(sys.argv[3])
batch_training_size = int(sys.argv[4])

data_part = 5 # only one fifth of the whole dataset to speed up training

for i in range(len(labels_full) // batch_size // data_part):
    images.append(images_full[i*batch_size : (i+1)*batch_size])
    labels.append(labels_full[i*batch_size : (i+1)*batch_size])

def sigmoid_prime(x):
    return np.exp(-x) / ((np.exp(-x) + 1) ** 2)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# nowe, przyda się?
def relu(x):
    return np.maximum(x, x * 0.01)

def relu_prime(x):
    if x >= 0:
        return 1
    # ej nie jest tak xd
    # a jak xd?
    type(x) == no.ndarray
    # no x to macierz xd
    # np.exp jest przeładowane ale jakakoleiwk funkcja to chyba nie
    # to co foreach ? :(
    # właśnie nie wiem, a co z gpu?
    # to miało być szybsze a nie xd
    # mamy duzo mozliwosci zmian ale nie na raz trzeba ustalic jakos
    # hm TODO gpu TODO wincyj procent TODO gui gotowe
    # xd
    # tamto myliło hah
    # to co najpierw? :p
    # ssh daje wglad do basha tylko tak ?
    # nie, to jest taki fajny programik, byobu
    # i ten pasek na dole też jest z byobu
    # on udostepnia tylko basha ?
    # tak, ale basha multiplayer xd
    # szkoda że 2 kursorow nie ma
    # hm

    return 0.01 # chyba tak xd nikt nie widzial xd
    # ale x to macierz :p
# ale to jest przeciazone i jak jest funkcja od macierzy to bierze po kolei kazdy element
# w sumie
# zobacze na drugiej karcie xd
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
SIZES = [ 784, size_1, size_2, 10 ]

syn0 = 2 * np.random.random((SIZES[0], SIZES[1])) - 1  
syn1 = 2 * np.random.random((SIZES[1], SIZES[2])) - 1 
syn2 = 2 * np.random.random((SIZES[2], SIZES[3])) - 1  

# biases for respective layers
b0 = 2 * np.random.random((1, SIZES[1])) - 1
b1 = 2 * np.random.random((1, SIZES[2])) - 1
b2 = 2 * np.random.random((1, SIZES[3])) - 1

for i, batch in list(enumerate(images)):
    X = np.array(batch)
    print("x:")
    print(np.shape(X))
    
    print("======================= BATCH {} =======================".format(i))

    error = 1
    j = 0
    while j < batch_training_size:
        l0 = X
        l1 = sigmoid(np.dot(l0, syn0) + b0)
        l2 = sigmoid(np.dot(l1, syn1) + b1)
        l3 = sigmoid(np.dot(l2, syn2) + b2)

        l3_error = (y[i] - l3)#** 2

        error = np.mean(np.abs(l3_error))

        j += 1
        if j % 20 == 0:
            print(("[%d] error: " % j) + str(error))

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
    correct_rate = correct / (i + 1.0)
    print("{} = {} (correct {}%)".format(label, prediction, 100 * correct_rate))

with open('log/' + str(datetime.now()), 'a') as f:
    with open(__file__, 'r') as myself:
        print(myself.read(), file=f)
    print("", file=f)
    print("#### answers:", file=f)
    print("argv =", sys.argv, file=f)
    print("correct_rate =", correct_rate, file=f)
    print("SIZES =", SIZES, file=f)
    print("syn0 =", syn0, file=f)
    print("syn1 =", syn1, file=f)
    print("syn2 =", syn2, file=f)
    print("b0 =", b0, file=f)
    print("b1 =", b1, file=f)
    print("b2 =", b2, file=f)
