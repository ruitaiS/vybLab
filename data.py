import pickle
import numpy as np

'''
(E)MNIST Full Set:
    60k training digits
    60k training letters
    10k testing digits
    10k testing letters

Subnet Training Set (digits):
    40k digits only (from training digits)

Alternet Training Set (letters):
    40k letters only (from training letters)

Supernet Training Set (mixed):
    40k mixed (remaining 20k from digits, 20k from letters)

MetaNet Test Set (Mixed):
    20k mixed (10k digits test, 10k letters test)

Return Format: 
    Returns a shuffled list of datums (or a list of lists of datums if we want to subdivide each set)
    Datum: (img, label) pair or (img, label, meta_label) tuple
        label = 0-9 for digits; 10-35 for letters
        meta_label = 0 if digit, 1 if letter

When testing only pass the img portion to the NN; tester sees the label / metalabel

'''


def split(dataset, no_of_pieces):
    #dataset = shuffle(dataset)

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

#TODO: Do we really need to keep the existing training / test distinction?
#TODO: not happy w/ how this assigns the data to groups before shuffling; should shuffle *all* and then group
class Data:
    def __init__(self):

        #TODO: I think this could be optimized for better space efficiency (rn we're keeping the whole dataset in memory)
        with open("data/mnist/pickled_mnist.pkl", "br") as fh:
            data = pickle.load(fh)

        digits_train_imgs = data[0]
        digits_train_labels = data[2]
        digits_test_imgs = data[1]
        digits_test_labels = data[3]

    
        with open("data/emnist/pickled_emnist.pkl", "br") as fh:
            data = pickle.load(fh)

        letters_train_imgs = data[0]
        letters_train_labels = data[2]
        letters_test_imgs = data[1]
        letters_test_labels = data[3]

        #Increment letter labels by 10
        #TODO: This will not work with NN oneHotting
        # (will put labels out of range of NN output array size)
        #Need to convert back or find some other workaround for that
        letters_train_labels = [i + 10 for i in letters_train_labels]
        letters_test_labels = [i + 10 for i in letters_test_labels]

        #TODO: Check the sizes of these after init to make sure the subarrays aren't being garbage collected
        self.subNet_train = list(zip(digits_train_imgs[:40000], np.array(digits_train_labels[:40000])))
        self.alterNet_train = list(zip(letters_train_imgs[:40000], np.array(letters_train_labels[:40000])))

        self.metaNet_train = list(zip(
            np.concatenate((digits_train_imgs[40000:], letters_train_imgs[40000:])) , 
            np.concatenate((digits_train_labels[40000:], letters_train_labels[40000:])) , 
            np.concatenate((np.full(20000, 0), np.full(20000, 1))) 
            ))

        self.metaNet_test = list(zip(
                np.concatenate((digits_test_imgs, letters_test_imgs)) , 
                np.concatenate((digits_test_labels, letters_test_labels)) ,
                np.concatenate((np.full(10000, 0),np.full(10000,1)))
                ))

    #TODO: rename to something less obtuse
    def subNet_Trainset(self, no_of_pieces):
        return split(np.random.shuffle(self.subNet_train),no_of_pieces)
    def alterNet_Trainset(self, no_of_pieces):
        return split(np.random.shuffle(self.alterNet_train),no_of_pieces)
    def metaNet_Trainset(self, no_of_pieces):
        return split(np.random.shuffle(self.metaNet_train),no_of_pieces)
    def metaNet_Testset(self, no_of_pieces):
        return split(np.random.shuffle(self.metaNet_test),no_of_pieces)