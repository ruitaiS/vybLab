import numpy as np
from scipy.stats import truncnorm

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


class ClusterModel: 
    def __init__(self, number_of_labels=20):
        self.sub_net = NeuralNetwork(no_of_in_nodes = image_pixels, 
                        no_of_out_nodes = number_of_labels, 
                        no_of_hidden_nodes = 60,
                        learning_rate = 0.1)
        self.super_network = NeuralNetwork(no_of_in_nodes = number_of_labels, 
                        no_of_out_nodes = 2, 
                        no_of_hidden_nodes = 15,
                        learning_rate = 0.1)
        self.data_points = np.array([])
        self.subnet_confusion_matrix = np.zeros((2, 2)) #[Predicted, Actuial]

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
            return (np.argmax(result), 0) #TODO: Implement clustering algorithm as alternate method. 

    def run(self, data, label, meta_label): #meta: 0 if seen before, 1 otherwise.
        result = self.sub_net.run(data).flatten()
        meta_result = self.super_network.run(result)
        if np.argmax(meta_result) == 1: #We have seen it before, return the result of sub network. 
            return (np.argmax(result), 1)
        else:                           #we have not seen it before go to alternate calssification method
            return (np.argmax(result), 0) #TODO: Implement clustering algorithm as alternate method. 