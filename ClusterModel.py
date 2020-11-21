import numpy as np
from scipy.stats import truncnorm
import pickle
import pandas as pd
from scipy.cluster.vq import vq, kmeans2, whiten
from statistics import mode


#Parameters for the data we are working with. 
image_size = 28 # width and length
labels = [0,1,2,3,4,5,6,7,8,9]
image_pixels = image_size * image_size
data_path = "data/mnist/"

#function(NDArray) will apply the function to every element in the array & return a new one
def sigmoid(x):
        return 1 / (1 + np.e ** -x)
np.seterr(all='ignore')

def tanh(x): 
    return 0.5 * (np.tanh(x) + 1)

#The activation function for the model. 
activation_function = sigmoid

#Helper method to convert to one hot. 
def to_one_hot(n, size=10): 
    r = np.zeros(size)
    r[int(n)] = 1.0
    return r

def from_one_hot(v): 
    return np.argmax(v)

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

def euclidian_distance(a, b): 
        dist = np.linalg.norm(a-b)
        return dist

class Classifier:
    #Initialize the clustering algorithm. 
    def __init__(self, one_hot_function = to_one_hot, one_hot_inverse_function = from_one_hot):
        #Initialize the list of known points. 
        self.data_points = []
        self.one_hot_function = one_hot_function
        self.one_hot_inverse_function = one_hot_inverse_function
    
    def train(self, input_vector, target_vector): 
        #target_vector = self.one_hot_inverse_function(target_vector) #Convert the label from one hot. UNCOMMENT THIS IF THE INPUT IS A VECTOR. 
        t = self.run(input_vector) #Run the input vector through the model before we add it. 
        #Add this vector to the database. 
        entry = [target_vector] + list(input_vector)
        self.data_points.append(entry)
        return t

    def run(self, input_vector):
        #Only run the clustering algorithm if we have enough data. 
        if len(self.data_points) == 0: 
            return None
        #Get the k nearest neighbors. 
        labs =  [i[0] for i in self.data_points]
        points = [i[1:] for i in self.data_points]
        centroids, labels  = kmeans2(points, len(labs), minit='points')
        #Take return the nearest mean.
        res = np.argmax([euclidian_distance(input_vector, i) for i in centroids])
        #Find the label that belongs to that centroid. 
        candidates = []
        for (i, j) in zip(self.data_points, labels): 
            if j == res: 
                candidates.append(i[0])
        return mode(candidates)

class ClusterModel: 
    def __init__(self, number_of_labels=37):
        self.sub_net = NeuralNetwork(no_of_in_nodes = image_pixels, 
                        no_of_out_nodes = number_of_labels, 
                        no_of_hidden_nodes = 60,
                        learning_rate = 0.1)
        self.super_network = NeuralNetwork(no_of_in_nodes = number_of_labels, 
                        no_of_out_nodes = 2, 
                        no_of_hidden_nodes = 15,
                        learning_rate = 0.1)
        self.subnet_confusion_matrix = np.zeros((2, 2)) #[Predicted, Actuial]
        self.clustering_algorithm = Classifier()

    #Train the sub networks. data is a list of ((image, is_char), lab) lab is an int
    #The subnetworks predict, and detect when we have something new to add to the clustering algorithm.
    #DO NOT USE THIS UNLESS YOU SPECIFICALLY NEED TO, you can just use train and run below. 
    def train_sub_networks(self, data):
        print("Trainig sub-networks.")
        for d in data: 
            ((img, is_char), lab) = d
            if is_char: 
                t = self.sub_net.run(img)
                s = self.super_network.predict(t.flatten(), [0.0, 1.0])
                #Update the confusion matrix. 
                if np.argmax(s) == 1: 
                    self.subnet_confusion_matrix[1, 1] += 1
                else: 
                    self.subnet_confusion_matrix[0, 1] += 1
            else: 
                t = self.sub_net.predict(img, to_one_hot(lab))
                s = self.super_network.predict(t.flatten(), [1.0, 0.0])
                #Update the confusion matrix. 
                if np.argmax(s) == 1: 
                    self.subnet_confusion_matrix[1, 0] += 1
                else: 
                    self.subnet_confusion_matrix[0, 0] += 1
        l = len(data)
        self.subnet_confusion_matrix = np.array([x/l for x in self.subnet_confusion_matrix.reshape(4)]).reshape((2,2))
        print("Done training, Confusion Matrix:")
        print(self.subnet_confusion_matrix)

    #Train the model with the bit of input data. Returns prediction before training. 
    def train(self, data, label, meta_label):  #meta: 0 if seen before, 1 otherwise.
        result = self.sub_net.run(data).flatten()
        meta_result = self.super_network.run(result)
        self.sub_net.train(data, to_one_hot(label, size=self.sub_net.no_of_out_nodes))
        self.super_network.train(result, to_one_hot(meta_label, size=self.super_network.no_of_out_nodes))
        if np.argmax(meta_result) == 1: #We have seen it before, return the result of sub network. 
            return (np.argmax(result), 1)
        else:                           #we have not seen it before go to alternate calssification method
            return (self.clustering_algorithm.train(data, label), 0) 

    def run(self, data, label, meta_label): #meta: 0 if seen before, 1 otherwise.
        result = self.sub_net.run(data).flatten()
        meta_result = self.super_network.run(result)
        if np.argmax(meta_result) == 1: #We have seen it before, return the result of sub network. 
            return (np.argmax(result), 1)
        else:                           #we have not seen it before go to alternate calssification method
            return (self.clustering_algorithm.run(data, label), 0)
