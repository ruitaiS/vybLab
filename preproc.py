#Based on code from here:
#https://www.python-course.eu/neural_network_mnist.php

import numpy as np
import matplotlib.pyplot as plt
import pickle

def process(data_path, train_file, test_file, output_path):
    train_data = np.loadtxt(data_path + train_file, 
                            delimiter=",")
    test_data = np.loadtxt(data_path + test_file, 
                           delimiter=",") 

    #Remap Data Values from [0-255] to [0.01-1]
    fac = 0.99 / 255
    train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
    test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01

    #Pull the labels (First element of every array)
    train_labels = np.asfarray(train_data[:, :1])
    test_labels = np.asfarray(test_data[:, :1])

    #Dump Everything to Pickle
    with open(output_path, "bw") as fh:
        data = (train_imgs, 
                test_imgs, 
                train_labels,
                test_labels)
        pickle.dump(data, fh)

process("data/mnist/", "mnist_train.csv", "mnist_test.csv", "data/mnist/pickled_mnist.pkl")
process("data/emnist/", "emnist_train.csv", "emnist_test.csv", "data/emnist/pickled_emnist.pkl")
