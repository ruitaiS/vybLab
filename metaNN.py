from data import Data
from NN import NeuralNet


class MetaNet:
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


    

    def generateChild(self, training_set):
        child = NeuralNet(no_of_in_nodes = image_pixels,

            #Output vector size is equal to vector size of current network
            #As we create new categories each "generation" of network will have more outnodes
            no_of_out_nodes = len(self.run(training_set[0])), 
            no_of_hidden_nodes = 100,
            learning_rate = 0.1)
        for i in range(0, len(training_set)):
            #Child sees the training image, but the parent network decides what the label should be
            #child network never actually sees the 'real' label (and neither does the parent)
            child.train(training_set[i], np.argmax(self.run(training_set[i])))
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