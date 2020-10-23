import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import math
from scipy.stats import truncnorm

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
        A method to initialize the weight 
        matrices of the neural network
        """
        rad = 1 / np.sqrt(self.no_of_in_nodes)
        X = truncated_normal(mean=0, 
                             sd=1, 
                             low=-rad, 
                             upp=rad)
        self.wih = X.rvs((self.no_of_hidden_nodes, 
                                       self.no_of_in_nodes))
        rad = 1 / np.sqrt(self.no_of_hidden_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.who = X.rvs((self.no_of_out_nodes, 
                                         self.no_of_hidden_nodes))
        
    
    def train(self, input_vector, target_vector):
        """
        input_vector and target_vector can 
        be tuple, list or ndarray
        """
        
        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T
        
        output_vector1 = np.dot(self.wih, 
                                input_vector)
        output_hidden = activation_function(output_vector1)
        
        output_vector2 = np.dot(self.who, 
                                output_hidden)
        output_network = activation_function(output_vector2)
        
        output_errors = target_vector - output_network
        # update the weights:
        tmp = output_errors * output_network \
              * (1.0 - output_network)     
        tmp = self.learning_rate  * np.dot(tmp, 
                                           output_hidden.T)
        self.who += tmp


        # calculate hidden errors:
        hidden_errors = np.dot(self.who.T, 
                               output_errors)
        # update the weights:
        tmp = hidden_errors * output_hidden * \
              (1.0 - output_hidden)
        self.wih += self.learning_rate \
                          * np.dot(tmp, input_vector.T)
        

    #Called by the testing framework. 
    def predict(self, input_vector, label): 
        res = self.run(input_vector)
        self.train(input_vector, label)
        return res
    
    def run(self, input_vector):
        # input_vector can be tuple, list or ndarray
        input_vector = np.array(input_vector, ndmin=2).T

        output_vector = np.dot(self.wih, 
                               input_vector)
        output_vector = activation_function(output_vector)
        
        output_vector = np.dot(self.who, 
                               output_vector)
        output_vector = activation_function(output_vector)
    
        return output_vector
            
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

#Load the data into memory. 
print("Loading data.")
train_imgs = pickle.load(open( data_path + 'train_imgs.p', "rb" ))
test_imgs = pickle.load(open( data_path + 'test_imgs.p', "rb" ))
train_labels = pickle.load(open( data_path + 'train_labels.p', "rb" ))
test_labels = pickle.load(open( data_path + 'test_labels.p', "rb" ))
print("Done ,loading data.")

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