#Based on this code:
#https://www.python-course.eu/neural_network_mnist.php

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

import random


import data
from NN import NeuralNetwork


#------------------------------------------------------

#Creates a Digit / Meta NN pair
#The Digit NN is trained on digits, then the Meta NN is trained on the outputs of that specific digit NN
def generate():
    Digit_NN = NeuralNetwork(no_of_in_nodes = image_pixels, 
                        no_of_out_nodes = 10, 
                        no_of_hidden_nodes = 100,
                        learning_rate = 0.1)
    '''
    Letter_NN = NeuralNetwork(no_of_in_nodes = image_pixels, 
                        no_of_out_nodes = 26, 
                        no_of_hidden_nodes = 100,
                        learning_rate = 0.1)
    '''

    Meta_NN = NeuralNetwork(Digit_NN.no_of_out_nodes, 2, 100, 0.05)

    #Train Digit NN
    for i in range(len(digits_train_imgs)):
        Digit_NN.train(digits_train_imgs[i], digits_train_labels_one_hot[i])

    #Display Statistics for Digits
    corrects, wrongs = Digit_NN.evaluate(digits_train_imgs, digits_train_labels)
    print("accuracy train: ", corrects / ( corrects + wrongs))
    corrects, wrongs = Digit_NN.evaluate(digits_test_imgs, digits_test_labels)
    print("accuracy: test", corrects / ( corrects + wrongs))

    #Train MetaNN, save NN output vectors to be evaluated later
    for i in range(len(mixed_train_imgs)):
        Meta_NN.train(np.sort(Digit_NN.run(mixed_train_imgs[i]).T), mixed_train_labels[i])

    #Display Statistics for Meta
    #TODO: Investigate whether this has redundant code
    corrects, wrongs = Meta_NN.metaEval(Digit_NN, mixed_train_imgs, mixed_train_labels)
    train_accuracy = corrects / ( corrects + wrongs)
    print("Train Accuracy: ", train_accuracy)
    print("Train Confusion Matrix: ")
    print(Meta_NN.meta_confusion_matrix(Digit_NN,mixed_train_imgs, mixed_train_labels, mixed_train_values))

    corrects, wrongs = Meta_NN.metaEval(Digit_NN, mixed_test_imgs, mixed_test_labels)
    test_accuracy = corrects / ( corrects + wrongs)
    print("Test Accuracy: ", test_accuracy)
    print("Test Confusion Matrix: ")
    print(Meta_NN.meta_confusion_matrix(Digit_NN,mixed_test_imgs, mixed_test_labels, mixed_test_values))

    return train_accuracy, test_accuracy, Digit_NN, Meta_NN

#Iterate 100 Times and Collect data
def test(iterations):
    train_sum = 0
    test_sum = 0
    for i in range(iterations):
        a, b, digit, meta = generate()
        train_sum += a
        test_sum += b
    print("Train Average: ", train_sum / iterations)
    print("Test Average: ", test_sum / iterations)

#test(100)

#Returns a list of high performing Meta_NN's 
def collectBest(minAccuracy, amount):
    metaNets = []
    subNets = []
    while (len(metaNets) < amount):
        a,b,letters, meta = generate()
        subNets.append(letters)
        if (a + b)/2 > minAccuracy:
            metaNets.append(meta, letters)
    return metaNets, subNets
    



#Roughly 0.5 without sorting
#0.7ish with sorting, sometimes dips into 0.4s and .3's, but usually .6+
#What is the deal with this variation??

'''    
for i in range(len(digits_train_imgs)):
    ANN.train(digits_train_imgs[i], digits_train_labels_one_hot[i])
for i in range(20):
    res = ANN.run(digits_test_imgs[i])
    print(digits_test_labels[i], np.argmax(res), np.max(res))

cm = ANN.confusion_matrix(digits_train_imgs, digits_train_labels)
print(cm)

for i in range(10):
    print("digit: ", i, "precision: ", ANN.precision(i, cm), "recall: ", ANN.recall(i, cm))
'''
