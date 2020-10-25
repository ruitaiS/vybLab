#Based on this code:
#https://www.python-course.eu/neural_network_mnist.php

import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import truncnorm

import random

from NN import NeuralNetwork

#Initializing Datasets:
#------------------------------------------------------
image_size = 28 # width and length
image_pixels = image_size * image_size

with open("data/mnist/pickled_mnist.pkl", "br") as fh:
    data = pickle.load(fh)

digits_train_imgs = data[0]
digits_test_imgs = data[1]
digits_train_labels = data[2]
digits_test_labels = data[3]
digits_train_labels_one_hot = data[4]
digits_test_labels_one_hot = data[5]

with open("data/emnist/pickled_emnist.pkl", "br") as fh:
    data = pickle.load(fh)

letters_train_imgs = data[0]
letters_test_imgs = data[1]
letters_train_labels = data[2]
letters_test_labels = data[3]
letters_train_labels_one_hot = data[4]
letters_test_labels_one_hot = data[5]

#TODO Ensure this way of splitting is methodologically sound

#Take last 1/3 of letters + digits training, make mixed training set
#40000 elements in each (letters, digits, mixed)
mixed_train_imgs = np.concatenate((digits_train_imgs[40000:], letters_train_imgs[40000:]))
mixed_train_labels = np.concatenate((np.full(20000, 0), np.full(20000, 1)))

#Take half of letters/digits test sets to make mixed test set
#10k each in originals; 10k in mixed set
mixed_test_imgs = np.concatenate((digits_test_imgs[5000:], letters_test_imgs[5000:]))
mixed_test_labels = np.concatenate((np.full(5000, 0), np.full(5000, 1)))

#Shuffle mixed images & labels so index matching is preserved
#TODO Confirm this actually does it properly
shuffler = np.random.permutation(len(mixed_train_imgs))
mixed_train_imgs = mixed_train_imgs[shuffler]
mixed_train_labels = mixed_train_labels[shuffler]

shuffler2 = np.random.permutation(len(mixed_test_imgs))
mixed_test_imgs = mixed_test_imgs[shuffler2]
mixed_test_labels = mixed_test_labels[shuffler2]

#Remove last 20k from both img and label for letters/digits
digits_train_imgs = digits_train_imgs[:40000]
letters_train_imgs = letters_train_imgs[:40000]
digits_train_labels = digits_train_labels[:40000]
letters_train_labels = letters_train_labels[:40000]

#Remove last 5k from test img/label sets for letters/digits
digits_test_imgs = digits_test_imgs[:5000]
letters_test_imgs = letters_test_imgs[:5000]
digits_test_labels = digits_test_labels[:5000]
letters_test_labels = letters_test_labels[:5000]

print(len(mixed_test_labels))
print(len(mixed_test_imgs))
#------------------------------------------------------


Digit_NN = NeuralNetwork(no_of_in_nodes = image_pixels, 
                    no_of_out_nodes = 10, 
                    no_of_hidden_nodes = 100,
                    learning_rate = 0.1)

Letter_NN = NeuralNetwork(no_of_in_nodes = image_pixels, 
                    no_of_out_nodes = 26, 
                    no_of_hidden_nodes = 100,
                    learning_rate = 0.1)

Meta_NN = NeuralNetwork(Digit_NN.no_of_out_nodes, 2, 100, 0.1)

'''
print(len(digits_train_labels))
print(type(digits_train_labels))
for i in range (20):
    print (digits_train_labels[i])

print(len(letters_train_labels))
for i in range (20):
    print (letters_train_labels[i])
'''

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
    #print(len(Digit_NN.run(mixed_train_imgs[i])))
    #TODO Figure out what is one_hot?
    Meta_NN.train(Digit_NN.run(mixed_train_imgs[i]).T, mixed_train_labels[i])

#Display Statistics for Meta
corrects, wrongs = Meta_NN.metaEval(Digit_NN, mixed_train_imgs, mixed_train_labels)
print("accuracy train: ", corrects / ( corrects + wrongs))
corrects, wrongs = Meta_NN.metaEval(Digit_NN, mixed_test_imgs, mixed_test_labels)
print("accuracy: test", corrects / ( corrects + wrongs))



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
