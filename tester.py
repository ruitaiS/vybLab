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

def test_model(model, plot_title="Accuracy Over Time", window_size=1000): 
    #Prepair the training data.
    data_path = "data/mnist/"
    char_path = "data/emnist/"

    print("Loading data.")
    #Load the data into memory. 
    num_train_imgs = pickle.load(open( data_path + 'train_imgs.p', "rb" ))
    num_train_labels = pickle.load(open( data_path + 'train_labels.p', "rb" ))
    char_train_imgs = pickle.load(open( char_path + 'emnist_train_imgs.p', "rb" ))
    char_train_labels = pickle.load(open( char_path + 'emnist_train_labels.p', "rb" ))

    char_train_labels = [i + 10 for i in char_train_labels] #Digits: 1-9, Characters:10-36 

    is_char = [1 for i in range(len(char_train_imgs))]
    is_num = [0 for i in range(len(num_train_imgs))]
    chars = list(zip(np.array(char_train_imgs), np.array(char_train_labels), is_char))
    numbs = list(zip(np.array(num_train_imgs)[:,1:], np.array(num_train_labels), is_num))
    combined_data = list(chars + numbs)
    random.shuffle(combined_data)
    random.shuffle(chars)
    random.shuffle(numbs)

    #data of the form (data, label, meta_label)

    #Initially train the model without any characters. 
    accuracy = []
    meta_accuracy = []
    print("Phase 1 training. (Only on characters.)")
    for d in chars[:200]:
        (img, lab, meta_lab) = d
        (result, meta_result) = model.train(img, lab, meta_lab)
        #Test the meta accuracy
        if np.argmax(meta_result) == meta_lab: 
            meta_accuracy.append(1)
        else: 
            meta_accuracy.append(0)
        #Test the subnet accuracy. 
        if np.argmax(result) == lab: 
            accuracy.append(1)
        else: 
            accuracy.append(0)
    
    #Add in the numbers and see if we can classify them. 
    print("Phase 2 training. (On both characters and numbers.)")
    for d in combined_data[-200:]:
        (img, lab, meta_lab) = d
        (result, meta_result) = model.train(img, lab, meta_lab)
        #Test the meta accuracy
        if np.argmax(meta_result) == meta_lab: 
            meta_accuracy.append(1)
        else: 
            meta_accuracy.append(0)
        #Test the subnet accuracy. 
        if np.argmax(result) == lab: 
            accuracy.append(1)
        else: 
            accuracy.append(0)

    #Graph the results. 
    print("Reporting Results")
    meta_accuracy = [sum(meta_accuracy[i:i+window_size])/window_size for i in range(len(meta_accuracy)-window_size)]
    accuracy = [sum(accuracy[i:i+window_size])/window_size for i in range(len(accuracy)-window_size)]

    plt.plot(accuracy, label="Model Accuracy")
    plt.plot(meta_accuracy, label="Meta Accuracy")
    plt.legend(loc='lower right')
    plt.title(plot_title)
    plt.show()

model = ClusterModel(number_of_labels=37, )
test_model(model, window_size=20)