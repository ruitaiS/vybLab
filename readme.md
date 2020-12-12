## Setup:

### Dependencies
	pip install numpy
	pip install matplotlib
	pip install scipy
	pip install pickle

### Installation
0. (Optional) Read the tutorial from [this website](https://www.python-course.eu/neural_network_mnist.php).

1. Download the MNIST [testing](https://www.python-course.eu/data/mnist/mnist_train.csv) and [training](https://www.python-course.eu/data/mnist/mnist_train.csv) datasets, and place them into the data/mnist folder.

2. Download the [EMNIST dataset](http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip).

    2.1 Extract the following files from the archive and place them into the data/emnist folder:

        emnist-letters-train-labels-idx1-ubyte
        emnist-letters-train-images-idx3-ubyte
        emnist-letters-test-labels-idx1-ubyte
        emnist-letters-test-images-idx3-ubyte

    2.2 Run convert.py to convert them into csv format

2. Run preproc.py (You only need to do this once).

3. Run codeTest.py

# Project Motivation
&nbsp;&nbsp;&nbsp;&nbsp; With traditional neural networks, inputs are classified according to a predetermined set of
possible outputs. While this is beneficial in situations where the categories are known ahead of
time and do not change, it does not allow for the formation of novel classifications. If an item
from a previously unseen category is presented, it will not be properly identified as something
new, but rather will be lumped into whichever existing category is the closest match.

&nbsp;&nbsp;&nbsp;&nbsp; In our project, we would like to develop a form of neural network which can identify novel categories and learn to classify items into those new categories over time. As proof of
concept, we will be focusing on identifying handwritten characters from the ​mnist handwritten
digit database​. Using supervised learning, we will first train a model that can reliably identify the numbers from 0-9, then we will give it letters from A-Z. When given letters, which it has never seen before, the NN should be able to (1) identify that they are not numbers, and (2) group the letters into clusters using unsupervised learning techniques.

# Algorithm Outline

### 1) Data Processing

<img src="Processing.png" align="left" alt="Data Processing"
	title="Data Processing"/>

Before we do anything, we need to process the data into a format that we can readily access. The MNIST dataset already comes to us as a CSV file, but the extended MNIST is actually a series of ubyte files, and needs to be converted into CSV using the convert.py script.

After the datasets are in CSV format, we run preproc.py to pickle them for faster access.

Finally data.py encapsulates the data into a class that we can then import into our main code. Getter functions within the Data class automatically shuffle or split up the data and returns them for use.

<br/><br/>
<br/><br/>
<br/><br/>
<br/><br/>
<br/><br/>
<br/><br/>
<br/><br/>
<br/><br/>
<br/><br/>

### 2) Execution Phases

<p align="center"><img src="Execution.png"  alt="Execution Phases"
	title="Execution Phases"/></p>

Within the metaNN class there are three seperate neural networks, all of which need to be trained.

The subnet and the alternet, which recognize digits and letters respectively, are trained first. The subnet is trained on a set of 40,000 digit image / label pairs, and similarly the alternet is trained on a set of 40,000 letter image / label pairs.

The supernet, which classifies the subnet's output as "valid" or "invalid," is trained next. It is important that the supernet is trained after the subnet, as we need to use the output vector from a fully trained subnet as the input for the supernet.


### 3) Child Network Generation
### 4) Graphing Results

# Project Findings / Results


# Code Documentation
## NN.py
The NN class is the foundation of the project; the metaNN is basically a wrapper containing three separate NN instances.

Each NN instance is a neural network with a single hidden layer. The size of the input, hidden, and output layers (as well as the learning rate) are parameterized; the weights between layers are randomized at initialization.

The networks use the sigmoid function as the activation function, and a truncated normal distribution to generate randomized weights.


*Note: A good amount of the code came from [this tutorial](https://www.python-course.eu/neural_network_mnist.php) - the methods which were not written by us are marked by asterisks. The explanations for what they do are correct as far as we know, but there is the possibility we've misunderstood something.*

### Helper Functions
	oneHot(int label, int list labelKey)
Converts an integer label to its one-hot encoding (a list where only the index element corresponding to the label has a high value, and all other indices have low values).

labelKey is the mapping from the label values to the indices, such that labelKey[i] contains the integer label which should be placed at the ith index in the one-hot vector.

	sigmoid(x)*
The activation function for nodes in the network. Used by the train() and run() methods to determine which nodes in the hidden layer / output layer should fire (?)

	truncated_normal(mean, sd, low, upp)*
Generates a truncated normal distribution. Used to assign random weights to the connecting edges when initializing a new NN instance. More detailed information can be found [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html).

### Class Functions
	__init__(number of input nodes, number of output nodes, number of hidden nodes, learning rate)*

Initializes a NN with the specified parameters. Note that it calls create_weight_matrices() to initialize the weights between nodes. It also produces a labelKey mapping based on the number of specified output nodes. For n output nodes, the default labelKey is a list of integers from 0 to n-1; the ith output node is labelled with value i. (see the section on the oneHot and set_label_key for more information on how the labelKey is used).

	create_weight_matrices()*
Creates randomized weight matrices between the input and hidden layer (self.wih) and between the hidden layer and the output layer (self.wih). 
The weights are random variates pulled from a truncated normal distribution.

	set_learning_rate(learning rate)
Allows user to set the learning rate after instantiation.

	set_label_key(label Key)
Allows user to set the labelKey list after instantiation.

This is useful if you want to label the output nodes as something other than their index position. For example, in the alterNet letter classifier, we wanted the output node labels to start at 10 instead of 0 (since 0-9 is already being used for numbers), so we set the labelKey to a list of integers from 10-25.

	train(input vector, label)
Trains the NN on a single input vector / integer label pairing. Returns the values of each output node as an output vector (these can be converted back into integer labels using the labelKey).

	run(input vector)
Runs the NN on a single input vector. Returns the values of each output node as an output vector (these can be converted back into integer labels using the labelKey).

## metaNN.py
The metanet consists of three different NN instances: a subnet, a supernet, and an alternet.

Every input is first passed to the subnet. The subnet is only trained to classify digits from 0 to 9, but it will still output a result even if it is passed a letter. The supernet takes the output vector from the subnet as input, and classifies the subnet's output as being valid (1), or invalid (0). If the supernet confirms that the subnet's output is valid, the metanet will return the result of the subnet. If however the supernet determines that the subnet's output is invalid (eg. the subnet has been passed a letter), then the metanet will run the input image through the alterNet instead.

(We planned for the the alterNet to be a clustering algorithm or some other classifier that is able to do unsupervised learning, but we never got around to implementing that side of things. As of now, the alterNet is simply another member of the NN class that is trained to recognize letters).

### Class Functions
	__init__(input vector size, subnet output vector size)
Generates a metaNet with the specified input and subnet output vector sizes. The number of supernet inputs is tied to the number of subnet outputs.

In our case, the input vector is of length 784, with each index representing the grayscale value of a pixel in a 28 by 28 square image. See the preproc.py section for more information on how the input images are processed into this format.

#### Setters
	setSubNet(self, subNet)
	setAlterNet(self, alterNet)
These two methods allow us to set the subnet and the alternet after instantiation

#### Trainers
	trainSubNet(self, img, label)
	trainAlterNet(self, img, label)
	trainSuperNet(self, img, img_label, super_label)
These methods run a single training instance through the subnet, alternet, and supernet. 

In addition to a label for the image, the supernet training method also includes a super_label parameter. This label is 0 if the image is a letter and 1 if the image is a digit.

See the section on the train() function in NN.py for more information on how the training functions work. 

Note that while the NN train and run functions return an output vector, with values at each index for how sure the network is that that index is the correct response, the metaNet train and run functions return an integer label.

	run(self, img)
Runs a single image through the metaNet, and returns an integer label.

	generateChild(self, training_set)
Takes a set of training data consisting of (image, label) pairs, and generates a child metaNN.

The input image is run through the parent network, which returns a label. This label is used to train the child's subnet. At no point does the parent or the child network ever see the "real" label - this is only used for statistics. 

The child's subnet imitates the entire functionality of the parent network - it's range of outputs encompasses both the parent's subnet outputs as well as the parent's alternet outputs.


## data.py
The data class loads the MNIST and extended MNIST datasets into memory.


(E)MNIST Full Set:
    60k training digits
    60k training letters
    10k testing digits
    10k testing letters

	split(dataset, no_of_pieces)
Splits the specified dataset into the specified number of subsets. Leftover elements are placed into the last set.

	__init__(self)
Loads mnist and emnist data from the pickled output from preproc.py. Note that labels for letters are offset by 10.

	shuffle(self)
Shuffles all the datasets.

#### Data Sets
Shuffles and returns the subnet training, alternet training, meta training, and meta testing sets respectively.

Returns a shuffled list of datapoints (or a list of lists of datapoints if we want to subdivide each set)

Each datapoint is an (img, label) pair or (img, label, meta_label) tuple.
The label is 0-9 for digits and 10-35 for letters.
The meta_label is 1 if the image is a digit, 0 if the image is a letter.
        
	sub_tr(self)
40k digits only (from training digits)

	alter_tr(self)
40k letters only (from training letters)

	meta_tr(self)
40k mixed (remaining 20k from digits, 20k from letters)

	meta_te(self)
20k mixed (10k digits test, 10k letters test)


## grapher.py
	 __init__(self)
	addGraph(self, accuracy, name)
	graph(self, graph)
	graphAll(self)
Contains functionality to show a simple accuracy plot over time.

## preproc.py
	process(data_path, train_file, test_file, output_path)
Takes the image pixel data csv files and pickles them for faster access.

Each line in the csv file has 785 entries. The first is the label for the image (a digit from 0-9), and the remaining 784 each represent the grayscale value from 0 to 255 of a single pixel in a 28 by 28 image. We place the labels into a separate list, and we re-scale the pixel values to range from 0.01 to 1 instead.

## convert.py*
	convert(imgf, labelf, outf, n)
This code was copied virtually unchanged from [this website](https://pjreddie.com/projects/mnist-in-csv/).

It takes the emnist training set and converts it into a series of csv files that are later processed by preproc.py.


## Concluding Remarks

![fin](lol.jpeg)

This was my first time documenting code, so hopefully it is helpful and not too vague or verbose. If you have any questions about how anything works please feel free to email me at shaoruitai@gmail.com.
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEzNDA4NTAxNTQsLTg0NTI5ODgyNywtMT
U3OTI2OTY4MywxNDA3MTg4NjAxXX0=
-->