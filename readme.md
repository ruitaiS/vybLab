## Setup:

0. (Optional) Read the tutorial from [this website](https://www.python-course.eu/neural_network_mnist.php)

1. Download the MNIST [testing](https://www.python-course.eu/data/mnist/mnist_train.csv) and [training](https://www.python-course.eu/data/mnist/mnist_train.csv) datasets, and place them into the data/mnist folder.

2. Download the [EMNIST dataset](http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip).

    2.1 Extract the following files and place them into the data/emnist folder:

        emnist-letters-train-labels-idx1-ubyte
        emnist-letters-train-images-idx3-ubyte
        emnist-letters-test-labels-idx1-ubyte
        emnist-letters-test-images-idx3-ubyte

    2.2 Run convert.py to convert them into csv format

2. Run preproc.py (You only need to do this once).

3. Run v1.py

## Algorithm Layout

(Re-read / re-write this after sleeping)

The MNIST dataset is a collection of handwritten digits from 0-9, stored as 28px by 28px images. The EMNIST dataset is a collection of handwritten *letters*, stored in the same format.

The training datasets contain 60,000 digits (or letters). The testing datasets contain 10,000 digits (or letters).

Our algorithm consists of a subnetwork which is trained to classify digit by looking at the grayscale values of the pixels of an input image, and a metanetwork which is trained to determine whether the subnetwork is certain of the output or not.

We pre-suppose that when the subnetwork is given a letter instead of a digit, it will be "uncertain" of the output. Because we've defined uncertainty in such a way, the meta-network is equivalent to a digit / letter classifier which uses the output vector from the subnetwork as input. However, as pointed out by Professor Vybihal, this is an imperfect distinction - for example, the subnetwork may be very sure that a "b" is actually a "6". 

The sub-network is first trained on 40,000 instances taken from the digit dataset alone. The meta-network is then trained on a mixed set consisting of the remaining 20,000 instances from the digit training dataset, plus an additional 20,000 instances from the letter dataset. This division of training data allows us to fully utilize the digit dataset without repetition while maintaining an equal number of training instances for both the sub and meta networks.

The subnet needs to be trained first independently, as it cannot be trained on a mixed set (there would be no proper label for characters), and training it only on numbers while having the meta train off it's output would not give the meta a meaningful training set, as it would not include any letters (we made this mistake ourselves)

*You could also have them train in parallel, but not update the subnetwork when feeding it a letter*

## Data.py

(Retype this later)

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

