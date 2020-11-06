import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import math
from scipy.stats import truncnorm
from NN import NeuralNetworkPipe
from ClusterModel import ClusterModel

#Parameters for the data we are working with. 
image_size = 28 # width and length
labels = [0,1,2,3,4,5,6,7,8,9]
image_pixels = image_size * image_size
data_path = "data/mnist/"

#The model definitions.
#Dummy model used for testing the tester. 
class DummyModel: 
    def __init__(self): 
        pass

    def predict(self, data, label): 
        return random.choice(labels)

#A epsilon greedy action selection. 
def sigmoid(x):
    return 1 / (1 + np.e ** -x)
activation_function = sigmoid

def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, 
                     (upp - mean) / sd, 
                     loc=mean, 
                     scale=sd)

def test_double_network(): 
    sub_model = NeuralNetworkPipe(no_of_in_nodes = image_pixels, 
                        no_of_out_nodes = 10, 
                        no_of_hidden_nodes = 60,
                        learning_rate = 0.1)

    model = NeuralNetworkPipe(no_of_in_nodes = 10, 
                        no_of_out_nodes = 2, 
                        no_of_hidden_nodes = 15, 
                        learning_rate = 0.1)

    model.assign_sub_network(sub_model)

    NeuralNetworkPipe.test_network(model)

def test_cluster_model(): 
    model = ClusterModel(10)

    #Prepair the training data.
    data_path = "data/mnist/"
    char_path = "data/emnist/"

    #Load the data into memory. 
    num_train_imgs = pickle.load(open( data_path + 'train_imgs.p', "rb" ))
    num_train_labels = pickle.load(open( data_path + 'train_labels.p', "rb" ))
    char_train_imgs = pickle.load(open( char_path + 'emnist_train_imgs.p', "rb" ))
    char_train_labels = pickle.load(open( char_path + 'emnist_train_labels.p', "rb" ))

    is_char = np.array(np.ones(len(char_train_imgs)), dtype='bool')
    is_num = np.array(np.zeros(len(num_train_imgs)), dtype='bool')
    chars = list(zip(zip(char_train_imgs, is_char), np.ones(len(char_train_imgs))))
    numbs = list(zip(zip(np.array(num_train_imgs)[:,1:], is_num), np.array(num_train_labels)))
    training_data = list(chars + numbs)
    random.shuffle(training_data)

    model.train_sub_networks(training_data)

test_cluster_model()