#Based on code from here:
#https://www.python-course.eu/neural_network_mnist.php

import numpy as np
import matplotlib.pyplot as plt
import pickle

def process(data_path, train_file, test_file, output_path):
    #Initialization / Loading Data into Arrays
    image_size = 28 # width and length
    no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
    image_pixels = image_size * image_size

    #These files are mondo yuge so it takes forever
    #Luckily we will pickle them so we only need to do this once
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

    #Converting to one-hot representation
    lr = np.arange(10)
    train_labels_one_hot = (lr==train_labels).astype(np.float)
    test_labels_one_hot = (lr==test_labels).astype(np.float)
    train_labels_one_hot[train_labels_one_hot==0] = 0.01
    train_labels_one_hot[train_labels_one_hot==1] = 0.99
    test_labels_one_hot[test_labels_one_hot==0] = 0.01
    test_labels_one_hot[test_labels_one_hot==1] = 0.99

    #Dump Everything to Pickle
    with open(output_path, "bw") as fh:
        data = (train_imgs, 
                test_imgs, 
                train_labels,
                test_labels,
                train_labels_one_hot,
                test_labels_one_hot)
        pickle.dump(data, fh)

process("data/mnist/", "mnist_train.csv", "mnist_test.csv", "data/mnist/pickled_mnist.pkl")
process("data/emnist/", "emnist_train.csv", "emnist_test.csv", "data/emnist/pickled_emnist.pkl")
