#Based on this code:
#https://www.python-course.eu/neural_network_mnist.php

import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import truncnorm
import random
import sys

#Determines activation output level of each node
#Layers are represented as NDArrays
#function(NDArray) will apply the function to every element in the array & return a new one
def sigmoid(x):
        return 1 / (1 + np.e ** -x)

def tanh(x): 
    return 0.5 * (np.tanh(x) + 1)

#The activation function for the model. 
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

#Helper method to convert to one hot. 
def to_one_hot(n): 
    r = np.zeros(10)
    r[int(n)] = 1.0
    return r

class NeuralNetworkPipe:
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

    #Assign a sub network to make predictions on. 
    def assign_sub_network(self, network): 
        self.sub_network = network 
    
    def train_subnetwork(self, input_vector, target_vector):
        return self.sub_network.train(input_vector, target_vector)
    
    def run_subnetwork(self, input_vector):
        return self.sub_network.run(input_vector)

    def predict_subnetwork(self, data, label): 
        return self.sub_network.predict(data, label)
    
    def train_supernetwork(self, input_vector, target_vector):
        return self.train(self.run_subnetwork(input_vector), target_vector)
    
    def train_both_networks(self, data, meta_label, label): 
        if np.argmax(meta_label) == 1: 
            #We have a char. 
            t = self.run_subnetwork(data)
            s = self.predict(t.flatten(), meta_label)
        else: 
            #We have a number. 
            t = self.predict_subnetwork(data, label)
            s = self.predict(t.flatten(), meta_label)
        return s
    
    def run_supernetwork(self, input_vector):
        return self.run(self.run_subnetwork(input_vector))

    def predict_supernetwork(self, data, label, meta_label): 
        t = self.run_subnetwork(data)
        return self.predict(t.flatten(), label) #Only run the subnet, we arent training it.

    #Make a prediction and train the network a little bit. 
    def predict(self, data, label): 
        t = self.run(data)
        self.train(data, label)
        return t
        
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

    #Test the network that this is given. 
    def test_network(super_net): 
        #Testing parameters. 
        image_size = 28 # width and length
        labels = [0,1,2,3,4,5,6,7,8,9]
        image_pixels = image_size * image_size
        data_path = "data/mnist/"
        char_path = "data/emnist/"

        #Load the data into memory. 
        print("Loading data.")
        num_train_imgs = pickle.load(open( data_path + 'train_imgs.p', "rb" ))
        #num_test_imgs = pickle.load(open( data_path + 'test_imgs.p', "rb" ))
        num_train_labels = pickle.load(open( data_path + 'train_labels.p', "rb" ))
        #num_test_labels = pickle.load(open( data_path + 'test_labels.p', "rb" ))

        char_train_imgs = pickle.load(open( char_path + 'emnist_train_imgs.p', "rb" ))
        #char_test_imgs = pickle.load(open( char_path + 'emnist_test_imgs.p', "rb" ))
        char_train_labels = pickle.load(open( char_path + 'emnist_train_labels.p', "rb" ))
        #char_test_labels = pickle.load(open( char_path + 'emnist_test_labels.p', "rb" ))
        
        print("Done loading data.")

        print("Bytes of data:", (sys.getsizeof(num_train_imgs) + 
            sys.getsizeof(num_train_labels) + 
            sys.getsizeof(char_train_imgs) + 
            sys.getsizeof(char_train_labels))/1000000, 
            "megabytes")

        print("Beginning training.")

        #Pre train the sub network. 
        split = len(num_train_imgs)
        num_accuracy_results = []
        for i in range(split):
            img = num_train_imgs[i][1:]
            lab = to_one_hot(num_train_labels[i])
            super_net.train_subnetwork(img, lab)

        #Stage two training. 
        is_char = np.array(np.ones(len(char_train_imgs)), dtype='bool')
        is_num = np.array(np.zeros(len(num_train_imgs)), dtype='bool')
        chars = list(zip(zip(char_train_imgs, is_char), np.ones(len(char_train_imgs))))
        numbs = list(zip(zip(np.array(num_train_imgs)[:,1:], is_num), np.array(num_train_labels)))
        vals = list(chars + numbs)
        random.shuffle(vals)
        for i in range(int(len(vals))): 
            ((img, is_char), lab) = vals[i]
            if is_char: 
                t = super_net.run_subnetwork(img)
                s = super_net.predict(t.flatten(), [0.0, 1.0])
            else:
                t = super_net.run_subnetwork(img)
                s = super_net.predict(t.flatten(), [1.0, 0.0])

        #Train both networks together.
        char_accuracy_results = []
        num_accuracy_results = []
        is_char = np.array(np.ones(len(char_train_imgs)), dtype='bool')
        is_num = np.array(np.zeros(len(num_train_imgs)), dtype='bool')
        chars = list(zip(zip(char_train_imgs, is_char), np.ones(len(char_train_imgs))))
        numbs = list(zip(zip(np.array(num_train_imgs)[:,1:], is_num), np.array(num_train_labels)))
        vals = list(chars + numbs)
        random.shuffle(vals)
        for i in range(int(len(vals))): 
            ((img, is_char), lab) = vals[i]
            if is_char: 
                t = super_net.run_subnetwork(img)
                s = super_net.predict(t.flatten(), [0.0, 1.0])
                if np.argmax(s) == 1: 
                    char_accuracy_results.append(1.0)
                else: 
                    char_accuracy_results.append(0.0)
                num_accuracy_results.append(None)
            else:
                t = super_net.predict_subnetwork(img, to_one_hot(lab))
                s = super_net.predict(t.flatten(), [1.0, 0.0])
                if np.argmax(s) == 0: 
                    char_accuracy_results.append(1.0)
                else: 
                    char_accuracy_results.append(0.0)
                if np.argmax(t) == lab: 
                    num_accuracy_results.append(1.0)
                else: 
                    num_accuracy_results.append(0.0)

        print("Done training.")
        print("Compiling results.")

        #Create a running average of the accuracy. 
        char_accuracy = []
        num_accuracy = []
        window = 1000
        for i in range(len(char_accuracy_results) - window):
            acc = 0
            for j in range(window): 
                acc += char_accuracy_results[i+j]
            char_accuracy.append(acc/window)
        for i in range(len(num_accuracy_results) - window):
            acc = 0
            count = 0
            for j in range(window): 
                if num_accuracy_results[i+j] != None: 
                    acc += num_accuracy_results[i+j]
                    count += 1
            num_accuracy.append(acc/count)
        plt.plot(char_accuracy, label="Meta Network Accuracy")
        plt.plot(num_accuracy, label="Sub Network Accuracy")
        plt.title("Staged Pre-Training 0.1 LR\nSigmoid Activation Function")
        plt.legend(loc="lower right")
        plt.show()