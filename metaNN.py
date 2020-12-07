import numpy as np
from data import Data
from NN import NeuralNet

class MetaNet:

    #TODO: Think about how to generalize this so that it can be built up
    def __init__(self, no_of_subnet_labels):
        self.subNet = NeuralNet(no_of_in_nodes = 28*28, 
                        no_of_out_nodes = no_of_subnet_labels, 
                        no_of_hidden_nodes = 60,
                        learning_rate = 0.1)

        self.superNet = NeuralNet(no_of_in_nodes = self.subNet.no_of_out_nodes, 
                        no_of_out_nodes = 2, 
                        no_of_hidden_nodes = 15,
                        learning_rate = 0.1)

        #TODO: This is a placeholder; it's supposed to grow as the alternet learns
        self.number_of_alternet_labels = 26
        #Placeholder alternate classifier
        self.alterNet = NeuralNet(no_of_in_nodes = 28*28, 
                        no_of_out_nodes = 26, 
                        no_of_hidden_nodes = 60,
                        learning_rate = 0.1)

        #Offset the alterNet labelKeys by the number of subNet labels so we don't overlap
        self.alterNet.set_label_key(np.array([i+no_of_subnet_labels for i in range(self.number_of_alternet_labels)]))

    #superNet is the core of the MetaNet instance, but sub and alter can be swapped out
    def setSubNet(self, subNet):
        self.subNet = subNet

    def setAlterNet(self, alterNet):
        self.alterNet = alterNet

    #Single Instance Training; returns predictions before altering weights
    def trainSubNet(self, img, label):
        return self.subNet.labelKey[np.argmax(self.subNet.train(img, label))]

    #TODO: May become defunct if alternet is designed to cluster (eg. learn unsupervised)
    def trainAlterNet(self, img, label):
        return self.alterNet.labelKey[np.argmax(self.alterNet.train(img, label))]

    #Train super with the bit of input data. 
    #Returns prediction as (img_label, meta_label) tuple
    def train(self, img, img_label, meta_label): 

        #Alternatively: Train subnet only if it's a digit
        subNet_outVector = self.subNet.run(img)
        metaNet_outVector = self.superNet.train(subNet_outVector.T, meta_label)

        #Return prediction result tuple
        #Use subnet prediction if super returns 1
        #else use alternet prediction
        '''
        if np.argmax(metaNet_outVector) == 1: 
            return (self.subNet.labelKey[np.argmax(subNet_outVector)], 1)
        else:
            return (self.alterNet.labelKey[np.argmax(self.alterNet.run(img))], 0)
        '''
        return np.argmax(metaNet_outVector)

    #Return result without altering weights
    #TODO: Should this be a tuple as well to keep it in line with train?
    def run(self, img):
        subNet_outVector = self.subNet.run(img)
        metaNet_outVector = self.superNet.run(subNet_outVector.T)
        if np.argmax(metaNet_outVector) == 1: 
            return self.subNet.labelKey[np.argmax(subNet_outVector)]
        else:
            return self.alterNet.labelKey[np.argmax(self.alterNet.run(img))]
    
    #TODO: Generalize this so it generates a MetaNet with a sub, meta, and alter net
    def generateChild(self, training_set):

        #Output vector size is equal to vector size of current network
        #As we create new categories each "generation" of network will have more outnodes
        child = MetaNet(no_of_subnet_labels = len(self.run(training_set[0])))

        wrong = 0
        total = len(training_set)

        for datum in training_set:
            img, label = datum
            parentResult = self.run(img)
            if (parentResult != label):
                wrong += 1
            child.subNet.train(img, parentResult)

        return child

    def getSubNet(self):
        return self.subNet

    def getAlterNet(self):
        return self.alterNet