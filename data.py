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
class Data:
    def __init__(self):

        #TODO: I think this could be optimized for better space efficiency (rn we're keeping the whole dataset in memory)
        with open("data/mnist/pickled_mnist.pkl", "br") as fh:
            data = pickle.load(fh)

        self.digits_train_imgs = data[0]
        self.digits_train_labels = data[2]
        self.digits_test_imgs = data[1]
        self.digits_test_labels = data[3]

    
        with open("data/emnist/pickled_emnist.pkl", "br") as fh:
            data = pickle.load(fh)

        self.letters_train_imgs = data[0]
        self.letters_train_labels = data[2]
        self.letters_test_imgs = data[1]
        self.letters_test_labels = data[3]

        #Increment letter labels by 10
        #TODO: This will not work with NN oneHotting
        # (will put labels out of range of NN output array size)
        #Need to convert back or find some other workaround for that

        self.letters_train_labels = np.array([i + 10 for i in self.letters_train_labels])
        self.letters_test_labels = np.array([i + 10 for i in self.letters_test_labels])

    def shuffle(self):
        #Shuffle Digits Train
        shuffler = np.random.permutation(len(self.digits_train_imgs))
        self.digits_train_imgs = self.digits_train_imgs[shuffler]
        self.digits_train_labels = self.digits_train_labels[shuffler]

        #Shuffle Digits Test
        shuffler = np.random.permutation(len(self.digits_test_imgs))
        self.digits_test_imgs = self.digits_test_imgs[shuffler]
        self.digits_test_labels = self.digits_test_labels[shuffler]
        
        #Shuffle Letters Train
        shuffler = np.random.permutation(len(self.letters_train_imgs))
        self.letters_train_imgs = self.letters_train_imgs[shuffler]
        self.letters_train_labels = self.letters_train_labels[shuffler]

        #Shuffle Letters Test
        shuffler = np.random.permutation(len(self.letters_test_imgs))
        self.letters_test_imgs = self.letters_test_imgs[shuffler]
        self.letters_test_labels = self.letters_test_labels[shuffler]
        

    #TODO: Do we really need to convert to a list then an np array?
    def sub_tr(self):
        self.shuffle()

        subNet_train = np.array(list(zip(self.digits_train_imgs[:40000], np.array(self.digits_train_labels[:40000]))))
        return subNet_train

    def alter_tr(self):
        self.shuffle()
        alterNet_train = np.array(list(zip(self.letters_train_imgs[:40000], np.array(self.letters_train_labels[:40000]))))
        return alterNet_train

    def meta_tr(self):
        self.shuffle()
        metaNet_train = np.array(list(zip(
            np.concatenate((self.digits_train_imgs[40000:], self.letters_train_imgs[40000:])) , 
            np.concatenate((self.digits_train_labels[40000:], self.letters_train_labels[40000:])) , 
            np.concatenate((np.full(20000, 0), np.full(20000, 1))) 
            )))

        shuffler = np.random.permutation(len(metaNet_train))
        metaNet_train = metaNet_train[shuffler]

        return metaNet_train

    def meta_te(self):
        self.shuffle()
        metaNet_test = np.array(list(zip(
                np.concatenate((self.digits_test_imgs, self.letters_test_imgs)) , 
                np.concatenate((self.digits_test_labels, self.letters_test_labels)) ,
                np.concatenate((np.full(10000, 0),np.full(10000,1)))
                )))

        shuffler = np.random.permutation(len(metaNet_test))
        metaNet_test = metaNet_test[shuffler]
        return metaNet_test