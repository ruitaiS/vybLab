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
    r[n] = 1.0
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
        #The initial training of the sub_model. (With only the numbers.)
        '''
        split = len(num_train_imgs)
        num_accuracy_results = []
        for i in range(split):
            img = num_train_imgs[i][1:]
            lab = to_one_hot(num_train_labels[i])
            if (np.argmax(super_net.predict_subnetwork(img, lab)) == np.argmax(lab)): 
                num_accuracy_results.append(1)
            else: 
                num_accuracy_results.append(0)
        '''
        #Compile the characters that we will be using.
        char_accuracy_results = []
        chars = list(zip(zip(char_train_imgs, np.ones(len(char_train_imgs))), np.ones(len(char_train_imgs))))
        numbs = list(zip(zip(np.array(num_train_imgs)[:,1:], np.zeros(len(num_train_imgs))), np.array(num_train_imgs)[:,1])) 
        vals = list(chars + numbs)
        #Recycle training data. 
        vals = vals + vals + vals
        random.shuffle(vals)
        for i in range(int(len(vals)/2)): 
            ((img, l), lab) = vals[i]
            meta_lab = [0,0]
            meta_lab[int(l)] = 1
            if(np.argmax(super_net.train_both_networks(img, meta_lab, lab)) == l):
                char_accuracy_results.append(1)
            else: 
                char_accuracy_results.append(0)

        print("Done training.")
        print("Compiling results.")

        #Create a running average of the accuracy. 
        accuracy = []
        window = 1000
        for i in range(len(char_accuracy_results) - window):
            acc = 0
            for j in range(window): 
                acc += char_accuracy_results[i+j]
            accuracy.append(acc/window)

        plt.plot(accuracy)
        plt.show()