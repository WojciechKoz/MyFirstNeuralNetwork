#!/usr/bin/env python3
# -*- coding: utf8 -*-
from __future__ import print_function # new print() on python2
from datetime import datetime
import sys
import argparse
import numpy as np  
from mnist import MNIST
from NeuralNetwork import Network

def arguments():
    parser = argparse.ArgumentParser(description='A simple neural network')
    parser.add_argument('--layer', type=int, action='append',
            help='add a layer with a given size to the network. Can be specified multiple times' +
            'to create multiple layers.')

    parser.add_argument('--batch-size', type=int, nargs=1,
            help='batch size to use during training')

    parser.add_argument('--batch-training-count', type=int, nargs=1,
            help='how many times to use backpropagation on any given batch')

    return parser.parse_args()


if __name__ == "__main__":
    arguments = arguments()

    # Display full arrays
    np.set_printoptions(threshold=np.inf)

    mndata = MNIST('./data')
    images_full, labels_full = mndata.load_training()
    images = []
    labels = []

    # dynamic arguments
    batch_size = arguments.batch_size[0]
    batch_training_size = arguments.batch_training_count[0]
    SIZES = [ 784 ] + arguments.layer + [ 10 ]
    print(SIZES)

    data_part = 5 # only one fifth of the whole dataset to speed up training

    for i in range(len(labels_full) // batch_size // data_part):
        images.append(images_full[i*batch_size : (i+1)*batch_size])
        labels.append(labels_full[i*batch_size : (i+1)*batch_size])

    y = []

    for batch in labels:
        y.append([])
        for label in batch:
            y[-1].append([1.0 if i == label else 0.0 for i in range(10)])

    y = np.array(y)


    network = Network(SIZES)

    network.epoch(images, y, batch_training_size)  # from dynamic parameters


    '''
    for i, el in enumerate(l3):
        print(labels[0][i], "=", np.argmax(el), " predictions: ", el)
    '''


    testing_images, testing_labels = mndata.load_testing()
    correct = 0.0
    for i, (image, label) in enumerate(zip(testing_images, testing_labels)):
        prediction = network.run(image)
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
        print("syn0 =", network.syn[0], file=f)
        print("syn1 =", network.syn[1], file=f)
        print("syn2 =", network.syn[2], file=f)
        print("b0 =", network.b[0], file=f)
        print("b1 =", network.b[1], file=f)
        print("b2 =", network.b[2], file=f)
