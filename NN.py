#Based on this code:
#https://www.python-course.eu/neural_network_mnist.php

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm


def oneHot(label, labelKey):
    oH = np.full(len(labelKey),0.01)
    for i in range(len(labelKey)):
        if label == labelKey[i]:
            oH[i] = 0.99
            break
    return oH

def sigmoid(x):
    #Determines activation output level of each node
    #Layers are represented as NDArrays
    #function(NDArray) will apply the function to every element in the array & return a new one
    return 1 / (1 + np.e ** -x)
activation_function = sigmoid

def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, 
                     (upp - mean) / sd, 
                     loc=mean, 
                     scale=sd)

class NeuralNet:
    
    def __init__(self, 
                 no_of_in_nodes, 
                 no_of_out_nodes, 
                 no_of_hidden_nodes,
                 learning_rate):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate 
        self.create_weight_matrices()

        #Default mapping for labels to one-hot indices
        #See oneHot() for use
        self.labelKey = np.array([i for i in range(no_of_out_nodes)])
        
    def create_weight_matrices(self):
        rad = 1 / np.sqrt(self.no_of_in_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)

        #WIH == Weight of Input layer to to Hidden layer(?)
        self.wih = X.rvs((self.no_of_hidden_nodes, self.no_of_in_nodes))
        
        rad = 1 / np.sqrt(self.no_of_hidden_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)

        #WHO == Weight of Hidden layer to Output layer(?)
        self.who = X.rvs((self.no_of_out_nodes, self.no_of_hidden_nodes))

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_label_key(self, lK):
        self.labelKey = lK         
    
    def train(self, input_vector, label):
        """
        input_vector and target_vector can 
        be tuple, list or ndarray
        """

        target_vector = oneHot(label, self.labelKey)
        
        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T
        
        output_vector1 = np.dot(self.wih, input_vector)
        output_hidden = activation_function(output_vector1)
        
        output_vector2 = np.dot(self.who, output_hidden)
        output_network = activation_function(output_vector2)
        
        output_errors = target_vector - output_network


        # update the weights:
        tmp = output_errors * output_network * (1.0 - output_network)     
        tmp = self.learning_rate  * np.dot(tmp, output_hidden.T)
        self.who += tmp

        # calculate hidden errors:
        hidden_errors = np.dot(self.who.T, output_errors)
        # update the weights:
        tmp = hidden_errors * output_hidden * (1.0 - output_hidden)
        self.wih += self.learning_rate * np.dot(tmp, input_vector.T)

        return output_network

    def run(self, input_vector):

        # input_vector can be tuple, list or ndarray
        input_vector = np.array(input_vector, ndmin=2).T

        output_vector = np.dot(self.wih, input_vector)
        output_vector = activation_function(output_vector)
        
        output_vector = np.dot(self.who, output_vector)
        output_vector = activation_function(output_vector)
    
        return output_vector

    def equals(self, NN):
        return ((np.equal(self.wih, NN.wih)) and (np.equal(self.who, NN.who)))



'''
create_weight_matrices generates a randomized weight matrix, and is called at initialization

set_learning_rate allows you to modify the learning rate after the NN is already initialized
    -> you also specify a learning rate when you first initialize the NN

train and run operate on a single input (plus label, in the case of train)
run is the only one that gives an output vector

labelKey[np.argmax(output_vector)] converts the output vector back into a label
'''