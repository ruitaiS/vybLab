import pickle
import numpy as np

'''
(E)MNIST Full Set:
    70k digits
    70k letters

Subnet Training Set (digits):
Alternet Training Set (letters):

Supernet Training Set (mixed):
Child Train Set (Mixed):
Child Test Set (Mixed):

Return Format: 
    Returns a shuffled list of datums (or a list of lists of datums if we want to subdivide each set)
    Datum: (img, label) pair or (img, label, meta_label) tuple
        label = 0-9 for digits; 10-35 for letters
        meta_label = 0 if digit, 1 if letter

When testing only pass the img portion to the NN; tester sees the label / metalabel

'''




class Data:
    def __init__(self):
        '''
        #NOTE: The original datasets had specific training / testing datasets, which we have merged here into a single set.
        #data(Old Version).py in the "old or unused" folder maintains the original training / testing distinction

        #TODO: I think this could be optimized for better space efficiency (rn we're keeping the whole dataset in memory)
        '''

        #Generates list of (img, label) pairs, one list for digits and one list for letters

        with open("data/mnist/pickled_mnist.pkl", "br") as fh:
            data = pickle.load(fh)
        digits_imgs = np.concatenate(data[0], data[1])
        digits_labels = np.concatenate(data[2], data[3])

        self.digits = np.array(list(zip(digits_imgs, digits_labels)))
    
        with open("data/emnist/pickled_emnist.pkl", "br") as fh:
            data = pickle.load(fh)
        letters_imgs = np.concatenate(data[0], data[1])
        letters_labels = np.concatenate(data[2], data[3])
        #Increment letter labels by 10
        letters_labels = np.array([i + 10 for i in letters_labels])

        self.letters = np.array(list(zip(letters_imgs, letters_labels)))

    def shuffleSet(self, inputSet):
        shuffler = np.random.permutation(len(inputSet))
        inputSet = inputSet[shuffler]

    def split(self, dataset, no_of_pieces):
        if no_of_pieces == 0:
            return dataset
        elif no_of_pieces == 1:
            return [dataset]

        remainder = len(dataset) % no_of_pieces
        chunkSize = int((len(dataset) - remainder) / no_of_pieces)

        result = []

        for i in range (0, len(dataset), chunkSize):
            chunk = []
            if (i + 2*chunkSize > len(dataset)):
                #print("Start: " + str(i) + "; End: " + str(len(dataset)-1))
                #print("Size: " + str(len(chunk)))
                chunk = dataset[i:len(dataset)]
                result.append(chunk)
                break
            else:
                #print("Start: " + str(i) + "; End: " + str(i+chunkSize-1))
                #print("Size: " + str(len(chunk)))
                chunk = dataset[i: i+chunkSize]
                result.append(chunk)

        print("Total Pieces: " + str(len(result)))
        return result        

    def shuffle(self):
        self.shuffleSet(self.digits)
        self.shuffleSet(self.letters)

    def assign(self):
        self.shuffle()

        split_digits = self.split(self.digits, 5)
        split_letters = self.split(self.letters, 5)

        #Digits Only
        self.sub_tr = np.concatenate(split_digits[0], split_digits[1])

        #Letters Only
        self.alter_tr = np.concatenate(split_letters[0], split_letters[1])

        #Mixed Digits / Letters, with super_label
        superlabels = np.concatenate((np.full(len(split_letters[2]), 0),np.full(len(split_digits[2]),1)))
        self.super_tr = np.array(list(zip(np.concatenate(split_letters[2], split_digits[2]) , superlabels)))
        self.shuffleSet(self.super_tr)
        
        #Mixed Digits / Letters
        self.child_tr = np.concatenate(split_letters[3], split_digits[3])
        self.shuffleSet(self.child_tr)

        self.child_te = np.concatenate(split_letters[4], split_digits[4])
        self.shuffleSet(self.child_te)