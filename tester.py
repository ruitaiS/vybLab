import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import math
from scipy.stats import truncnorm

#TODO: These imports don't work/exist
from metaNN import MetaNet

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
    
    #Record accuracies
    #It needs to be in an array for the grapher
    #accuracy[i] = 1 if data[i] was correctly classified; else 0
    accuracy = []
    meta_accuracy = []

    #TODO: When you train only on numbers, the meta never has an instance where it needs to output 1 (eg. that it's a character)
    #This is a problem.
    print("Phase 1 training. (Only on Numbers.)")
    for data in nums:
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
    
    #TODO: Half of this data (the numbers) are reused from phase 1
    print("Phase 2: Testing. (On both characters and numbers.)")
    for data in combined_data:
        (img, lab, meta_lab) = data
        (result, meta_result) = model.run(img, lab, meta_lab)

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
model = MetaNet()
test_model(model)