import numpy as np
from data import Data
from NN import NeuralNet

#Helper function for generateChild
#converts (integer) label into one-hot format
def oneHot(label, no_categories):
    res = np.full(no_categories,0.01)
    res[int(label)] = 0.99
    return res


class MetaNet:

    #Implemented in Sean's Code
    '''
    #assumes that the subnetwork is pre-trained already
    def __init__(self, subNet):
        self.subNet = subNet
        self.metaNet = NeuralNet(subNet.no_of_out_nodes, 2, 100, 0.2)


    def setSubNet(self, subNet):
        self.subNet = subNet

    def train(self, input_vector, target_vector):
    
    def trainSubNet(self, input_vector, target_vector):

    def run(self, input_vector):

    def equals(self, NN):
    '''

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
            child.train(training_set[i], oneHot(label, 10))

            #Store how much the parents gets wrong
            if (label != training_label[i]):
                wrong += 1
            total += 1

        print("Percentage Mislabelled: " + str(wrong/total))
        return child






    #------------------------------------------------------------------------------------------------------------------
    # Meta needs an associated subnet
    # 
    # 
    #------------------------------------------------------------------------------------------------------------------
    #Basically same kinda stuff as was in v1.py in the old version

    #subnet, metanet, alternet

    #takes subnet and alternet as initialization parameters?

    #init with default learning rate? and then have functions that can manually set the learning rates of the networks

    #run & test as endpoints for Sean
        #May need some adjusting of the data for this one, since meta has slightly different label format