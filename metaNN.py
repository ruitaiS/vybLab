import numpy as np
from data import Data
from NN import NeuralNet

class MetaNet:

    #TODO: Think about how to generalize this so that it can be built up
    def __init__(self, number_of_subnet_labels=10, number_of_alternet_labels = 26):
        self.subNet = NeuralNet(no_of_in_nodes = 28*28, 
                        no_of_out_nodes = number_of_subnet_labels, 
                        no_of_hidden_nodes = 60,
                        learning_rate = 0.1)
        self.superNet = NeuralNet(no_of_in_nodes = number_of_subnet_labels, 
                        no_of_out_nodes = 2, 
                        no_of_hidden_nodes = 15,
                        learning_rate = 0.1)

        #Placeholder alternate classifier for letters
        self.alterNet = NeuralNet(no_of_in_nodes = 28*28, 
                        no_of_out_nodes = number_of_alternet_labels, 
                        no_of_hidden_nodes = 60,
                        learning_rate = 0.1)

    #superNet is the core of the MetaNet instance, but sub and alter can be swapped out
    def setSubNet(self, subNet):
        self.subNet = subNet

    def setAlterNet(self, alterNet):
        self.alterNet = alterNet

    #Single Instance Training; returns predictions before altering weights
    def trainSubNet(self, img, label):
        return np.argmax(self.subNet.train(img, label))

    #TODO: May become defunct if alternet is designed to cluster (eg. learn unsupervised)
    def trainAlterNet(self, img, label):
        return np.argmax(self.alterNet.train(img, label))

    #Train super with the bit of input data. 
    #Returns prediction as (img_label, meta_label) tuple
    def train(self, img, img_label, meta_label): 

        #Alternatively: Train subnet only if it's a digit
        subNet_outVector = self.subNet.run(img)
        metaNet_outVector = self.superNet.train(subNet_outVector, meta_label)

        #Return prediction result tuple
        #Use subnet prediction if super returns 1
        #else use alternet prediction
        if np.argmax(metaNet_outVector) == 1: 
            return (np.argmax(subNet_outVector), 1)
        else:
            return (np.argmax(self.alterNet.run(img)), 0)

    #Return result without altering weights
    def run(self, img):
        subNet_outVector = self.subNet.run(img).flatten()
        metaNet_outVector = self.superNet.run(subNet_outVector)
        if np.argmax(metaNet_outVector) == 1: 
            return np.argmax(subNet_outVector)
        else:
            return np.argmax(self.alterNet.run(img))    
    
    #TODO: Generalize this so it generates a MetaNet with a sub, meta, and alter net
    def generateChild(self, training_set, training_label):
        child = NeuralNet(no_of_in_nodes = 28*28,
            #Output vector size is equal to vector size of current network
            #As we create new categories each "generation" of network will have more outnodes
            no_of_out_nodes = len(self.run(training_set[0])), 
            no_of_hidden_nodes = 100,
            learning_rate = 0.1)

        wrong = 0
        total = 0
        for i in range(0, len(training_set)):
            #Child sees the training image, but the parent network decides what the label should be
            #child network never actually sees the 'real' label (and neither does the parent)
            label = np.argmax(self.run(training_set[i]))
            child.train(training_set[i], label)

            #Store how much the parents gets wrong
            if (label != training_label[i]):
                wrong += 1
            total += 1

        print("Percentage Mislabelled: " + str(wrong/total))
        return child