import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import math
from scipy.stats import truncnorm
from NN import NeuralNetworkPipe

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
'''
#Load the data into memory. 
print("Loading data.")
train_imgs = pickle.load(open( data_path + 'train_imgs.p', "rb" ))
test_imgs = pickle.load(open( data_path + 'test_imgs.p', "rb" ))
train_labels = pickle.load(open( data_path + 'train_labels.p', "rb" ))
test_labels = pickle.load(open( data_path + 'test_labels.p', "rb" ))
print("Done loading data.")

print("Beginning testing.")

def to_one_hot(n): 
    r = np.zeros(10)
    r[n] = 1.0
    return r

#The model we will be testing. 
model = NeuralNetwork(no_of_in_nodes = image_pixels, 
                    no_of_out_nodes = 10, 
                    no_of_hidden_nodes = 100,
                    learning_rate = 0.1)

#Test the model over time. 
accuracy_results = []

#The initial training. (Without some lables.)
split = int(len(train_imgs)/2)
for i in range(split):
    img = train_imgs[i][1:]
    lab = to_one_hot(train_labels[i])
    if lab[-1] != 0.0 or lab[-2] != 0.0:  #Save 8 and 9 to introduce later. 
        continue
    if (np.argmax(model.predict(img, lab)) == np.argmax(lab)): 
        accuracy_results.append(1)
    else: 
        accuracy_results.append(0)

#The training with the extra label added. (With ALL labels. )
for i in range(split, len(train_imgs)): 
    img = train_imgs[i][1:]
    lab = to_one_hot(train_labels[i])
    if (np.argmax(model.predict(img, lab)) == np.argmax(lab)): 
        accuracy_results.append(1)
    else: 
        accuracy_results.append(0)

print("Done testing.")
print("Compiling results.")

#Create a running average of the accuracy. 
accuracy = []
window = 100
for i in range(len(accuracy_results) - window):
    acc = 0
    for j in range(window): 
        acc += accuracy_results[i+j]
    accuracy.append(acc/window)

plt.plot(accuracy)
plt.show()
'''
def test_double_network(): 
    sub_model = NeuralNetworkPipe(no_of_in_nodes = image_pixels, 
                        no_of_out_nodes = 10, 
                        no_of_hidden_nodes = 60,
                        learning_rate = 0.01)

    model = NeuralNetworkPipe(no_of_in_nodes = 10, 
                        no_of_out_nodes = 2, 
                        no_of_hidden_nodes = 20, 
                        learning_rate = 0.3)

    model.assign_sub_network(sub_model)

    NeuralNetworkPipe.test_network(model)

test_double_network()
