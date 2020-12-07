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

## NN.py
The NN class is the foundation of the project; the metaNN is basically a wrapper containing three separate NN instances.

Note: A good amount of the code came from [this tutorial](https://www.python-course.eu/neural_network_mnist.php) - the methods which were not written by us are marked by asterisks. The explanations for what they do are correct as far as we know, but there is the possibility we've misunderstood something.

### Helper Functions

#### oneHot(int label, int list labelKey)
Converts an integer label to its one-hot encoding (a list where only the index element corresponding to the label has a high value, and all other indices have low values).

labelKey is the mapping from the label values to the indices, such that labelKey[i] contains the integer label which should be placed at the ith index in the one-hot vector.

#### sigmoid(x)*
The activation function for nodes in the network. Used by the train() and run() methods to determine which nodes in the hidden layer / output layer should fire (?)

#### truncated_normal(mean, sd, low, upp)*
Generates a truncated normal distribution. Used to assign random weights to the connecting edges when initializing a new NN instance. More detailed information can be found [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html).

### Class Functions
#### __init__(number of input nodes, number of output nodes, number of hidden nodes, learning rate)*

Initializes a NN with the specified parameters. Note that it calls create_weight_matrices() to initialize the weights between nodes. It also produces a labelKey mapping based on the number of specified output nodes. For n output nodes, the default labelKey is a list of integers from 0 to n-1; the ith output node is labelled with value i. (see the section on the oneHot and set_label_key for more information on how the labelKey is used).

#### create_weight_matrices()*
Creates randomized weight matrices between the input and hidden layer (self.wih) and between the hidden layer and the output layer (self.wih). 
The weights are random variates pulled from a truncated normal distribution.

#### set_learning_rate(learning rate)
Allows user to set the learning rate after instantiation.

#### set_label_key(label Key)
Allows user to set the labelKey list after instantiation.

This is useful if you want to label the output nodes as something other than their index position. For example, in the alterNet letter classifier, we wanted the output node labels to start at 10 instead of 0 (since 0-9 is already being used for numbers), so we set the labelKey to a list of integers from 10-25.

#### train(input vector, label)
Trains the NN on a single input vector / integer label pairing. Returns the values of each output node as an output vector (these can be converted back into integer labels using the labelKey).

#### run(input vector)
Runs the NN on a single input vector. Returns the values of each output node as an output vector (these can be converted back into integer labels using the labelKey).

## metaNN.py
#### __init__(self, no_of_subnet_labels)
#### setSubNet(self, subNet)
#### setAlterNet(self, alterNet)
#### trainSubNet(self, img, label)
#### trainAlterNet(self, img, label)
#### train(self, img, img_label, meta_label)
#### run(self, img)
#### generateChild(self, training_set)
#### getSubNet(self)
#### getAlterNet(self)

## data.py
#### split(dataset, no_of_pieces)
#### __init__(self)
#### shuffle(self)
#### sub_tr(self)
#### alter_tr(self)
#### meta_tr(self)
#### meta_te(self)
## grapher.py
#### __init__(self)
#### addGraph(self, accuracy, name)
#### graph(self, graph)
#### graphAll(self)

## preproc.py
#### process(data_path, train_file, test_file, output_path)
## convert.py
#### convert(imgf, labelf, outf, n)

<!--stackedit_data:
eyJoaXN0b3J5IjpbMTc1ODA3NzEwNiwtMTM5NDA2Mzc5NiwtOT
YyMzA4NDU3LC03NTYwODI4NzQsLTE2MjgwNTM2MTAsNDg3MzUx
OTYwLC01MjE5MzM4NjEsLTE5NDYyMzExOSw5ODAzNzIxNjNdfQ
==
-->