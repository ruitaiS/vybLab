import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import math
from scipy.stats import truncnorm

#TODO: These imports don't work/exist
from NN import NeuralNetworkPipe
from ClusterModel import ClusterModel

from data import Data

#Dummy model used for testing the tester. 
class DummyModel: 
    labels = [0,1,2,3,4,5,6,7,8,9] 
    def __init__(self): 
        pass

    def predict(self, data, label):
        return random.choice(self.labels)

#TODO: Write this out
def updateConfMatrix(curr, data, result):
    #Takes the current confusion matrix, looks at the data the model was fed, and the result the model got back
    #Updates entries in confusion matrix and returns them
    return True


def test_model(model, plot_title="Accuracy Over Time", window_size=1000):
    '''
        model = metaNN
        TODO: This doesn't actually include the alternate classifier yet, only subnet and metaNet

        Loads the training sets for characters and numbers
        TODO: Refactor so it's in the data class, & you can just specify what kind of data you want

        


    '''



    print("Preparing data.")
    #Load the data into memory.
    data = Data()
    num_train_imgs = data.digits_train_imgs
    num_train_labels = data.digits_train_labels
    char_train_imgs = data.letters_train_imgs
    char_train_labels = data.letters_train_labels

    #Offset char labels so that 
    # Digits: 0-9, characters:10-36
    char_train_labels = [i + 10 for i in char_train_labels]  

    #Create a list of 1s for the characters, 0's for the numbers
    is_char = [1 for i in range(len(char_train_imgs))]
    is_num = [0 for i in range(len(num_train_imgs))]

    #TODO: Ask Sean
    #I think this creates a (img, label, is_char) tuple for each entry
    #Why np.array(char_train_labels)? (Maybe for later on?)
    #Also what is going on for num_train_imgs?
    chars = list(zip(char_train_imgs, np.array(char_train_labels), is_char))
    nums = list(zip(np.array(num_train_imgs)[:,1:], np.array(num_train_labels), is_num))

    combined_data = list(chars + nums)

    random.shuffle(combined_data)
    random.shuffle(chars)
    random.shuffle(nums)
    
    #Training Phase
    #TODO: Ask sean - No test only phase?
    
    #Record accuracies
    #It needs to be in an array for the grapher
    #accuracy[i] = 1 if data[i] was correctly classified; else 0
    accuracy = []
    meta_accuracy = []

    print("Phase 1 training. (Only on characters.)")
    for data in chars:
        #data of the form (data, label, meta_label)
        (img, lab, meta_lab) = data
        (result, meta_result) = model.train(img, lab, meta_lab)

        #Update the meta accuracy
        if np.argmax(meta_result) == meta_lab: 
            meta_accuracy.append(1)
        else: 
            meta_accuracy.append(0)

        #Update the subnet accuracy. 
        if np.argmax(result) == lab: 
            accuracy.append(1)
        else: 
            accuracy.append(0)
    
    print("Phase 2 training. (On both characters and numbers.)")
    for data in combined_data:
        (img, lab, meta_lab) = data
        (result, meta_result) = model.train(img, lab, meta_lab)

        #TODO: The results from both phases go into the same array?
        #Update the meta accuracy
        if np.argmax(meta_result) == meta_lab: 
            meta_accuracy.append(1)
        else: 
            meta_accuracy.append(0)
        #Update the subnet accuracy. 
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


#Initialize model and run tests
#TODO: Clustermodel doesn't exist; switch to metaNN
model = ClusterModel(number_of_labels=37)
test_model(model)