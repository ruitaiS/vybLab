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
        label = 0-9 for digits; 10-36 (or 37?) for letters
        meta_label = 0 if digit, 1 if letter

When testing only pass the img portion to the NN; tester sees the label / metalabel

'''


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


        #TODO: Check the sizes of these after init to make sure the subarrays aren't being garbage collected
        self.subNet_train = list(zip(digits_train_imgs[:40000], np.array(digits_train_labels[:40000])))
        self.alterNet_train = list(zip(letters_train_imgs[:40000], np.array(letters_train_labels[:40000])))
        


        
        

    def generateMixedSet(self):
        #TODO Ensure this way of splitting is methodologically sound

        #Labels are 0, 1 of whether something is a digit or letter
        #Values are the actual letter/digit being represented

        #Take last 1/3 of letters + digits training, make mixed training set
        #40000 elements in each (letters, digits, mixed)
        self.mixed_train_imgs = np.concatenate((self.digits_train_imgs[40000:], self.letters_train_imgs[40000:]))
        self.mixed_train_labels = np.concatenate((np.full(20000, 0), np.full(20000, 1)))
        self.mixed_train_values = np.concatenate((self.digits_train_labels[40000:], self.letters_train_labels[40000:]))



        #Take half of letters/digits test sets to make mixed test set
        #10k each in originals; 10k in mixed set
        self.mixed_test_imgs = np.concatenate((self.digits_test_imgs[5000:], self.letters_test_imgs[5000:]))
        self.mixed_test_labels = np.concatenate((np.full(5000, 0), np.full(5000, 1)))
        self.mixed_test_values = np.concatenate((self.digits_test_labels[5000:], self.letters_test_labels[5000:]))

        #Shuffle mixed images & labels so index matching is preserved
        #TODO Confirm this actually does it properly
        shuffler = np.random.permutation(len(self.mixed_train_imgs))
        self.mixed_train_imgs = self.mixed_train_imgs[shuffler]
        self.mixed_train_labels = self.mixed_train_labels[shuffler]
        self.mixed_train_values = self.mixed_train_values[shuffler]

        shuffler2 = np.random.permutation(len(self.mixed_test_imgs))
        self.mixed_test_imgs = self.mixed_test_imgs[shuffler2]
        self.mixed_test_labels = self.mixed_test_labels[shuffler2]
        self.mixed_test_values = self.mixed_test_values[shuffler2]

        #Remove last 20k from both img and label for letters/digits
        self.digits_train_imgs = self.digits_train_imgs[:40000]
        self.letters_train_imgs = self.letters_train_imgs[:40000]
        self.digits_train_labels = self.digits_train_labels[:40000]
        self.letters_train_labels = self.letters_train_labels[:40000]

        #Remove last 5k from test img/label sets for letters/digits
        self.digits_test_imgs = self.digits_test_imgs[:5000]
        self.letters_test_imgs = self.letters_test_imgs[:5000]
        self.digits_test_labels = self.digits_test_labels[:5000]
        self.letters_test_labels = self.letters_test_labels[:5000]

    #TODO: Works for existing structure, but I think it asssumes dataset is ndarray
    def shuffle(self, dataset):
        shuffler = np.random.permutation(len(dataset))
        return dataset[shuffler]

    #TODO: Splits the specified dataset into the specified number of pieces
    def split(self, dataset, no_of_pieces):



    def subNet_Trainset():
    def alterNet_Trainset():
    def superNet_Trainset():
    def metaNet_Testset():