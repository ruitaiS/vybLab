#Based on this code:
#https://www.python-course.eu/neural_network_mnist.php

import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import truncnorm

#Determines activation output level of each node
#Layers are represented as NDArrays
#function(NDArray) will apply the function to every element in the array & return a new one
def sigmoid(x):
    return 1 / (1 + np.e ** -x)
activation_function = sigmoid


'''
Truncated Normal Distribution Function
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html

truncnorm(a,b,loc, scale)
a,b >> the upper and lower bounds, calculated by:
    a = (desired lower bound - mean) / standard deviation
    b = (desired upper bound - mean) / standard deviation

loc >> shifts the distribution:
    Set it to the mean to center it

scale >> scales the distribution
'''
def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, 
                     (upp - mean) / sd, 
                     loc=mean, 
                     scale=sd)


class NeuralNetwork:
    
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
        
    def create_weight_matrices(self):
        """ 
        Initializes the weight 
        matrices of the neural network

        X is a truncated normal distribution,
        bounded between +/- 1/sqrt(# of input/hidden nodes)

        X.rvs(M,N) generates a 2D array (M arrays of N length each)
        from the normal distribution specified above
        
        Questions:
            - Why choose the bounds for the weights as a function of the # of nodes?
            - How exactly do the weights look in the structure of the NN?            
        
        """
        rad = 1 / np.sqrt(self.no_of_in_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)

        #WIH == Weight of Inner, Hidden Nodes??
        self.wih = X.rvs((self.no_of_hidden_nodes, self.no_of_in_nodes))
        
        rad = 1 / np.sqrt(self.no_of_hidden_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)

        #WHO == Weight of Hidden, Outer Nodes??
        self.who = X.rvs((self.no_of_out_nodes, self.no_of_hidden_nodes))
        
    
    def train(self, input_vector, target_vector):
        """
        input_vector and target_vector can 
        be tuple, list or ndarray
        """
        
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



    def run(self, input_vector):

        # input_vector can be tuple, list or ndarray
        input_vector = np.array(input_vector, ndmin=2).T

        output_vector = np.dot(self.wih, input_vector)
        output_vector = activation_function(output_vector)
        
        output_vector = np.dot(self.who, output_vector)
        output_vector = activation_function(output_vector)
    
        return output_vector

#---------------------------------------
#Rest of this stuff is for presentation:
#---------------------------------------
            
    #TODO For mixed sets, show which letters / numbers get confused for numbers / letters
    def confusion_matrix(self, data_array, labels):
        cm = np.zeros((10, 10), int)
        for i in range(len(data_array)):
            res = self.run(data_array[i])
            res_max = res.argmax()
            target = labels[i][0]
            cm[res_max, int(target)] += 1
        return cm    

    def precision(self, label, confusion_matrix):
        col = confusion_matrix[:, label]
        return confusion_matrix[label, label] / col.sum()
    
    def recall(self, label, confusion_matrix):
        row = confusion_matrix[label, :]
        return confusion_matrix[label, label] / row.sum()
        
    
    #This can't be used for Meta, because the input data is actually pre-processed by another NN first
    #Use metaEval instead
    def evaluate(self, data, labels):
        corrects, wrongs = 0, 0
        for i in range(len(data)):
            res = self.run(data[i])
            res_max = res.argmax()
            if res_max == labels[i]:
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs

    def metaEval(self, subNN, data, labels):
        corrects, wrongs = 0, 0
        for i in range(len(data)):
            res = self.run(np.sort(subNN.run(data[i]).T))
            res_max = res.argmax()
            if res_max == labels[i]:
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs
