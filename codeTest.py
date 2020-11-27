import numpy as np
from NN import NeuralNet
from metaNN import MetaNet
from data import Data, split
from grapher import Grapher

#convert (integer) label into one-hot format
#Used in metaNN

def oneHot(label, no_categories):
    res = np.full(no_categories,0.01)
    res[int(label)] = 0.99
    return res

def generateChild(parent, training_set, training_label, training_label_one_hot):
    data = Data()
    child1 = NeuralNet(no_of_in_nodes = 28*28,
        #Output vector size is equal to vector size of current network
        #As we create new categories each "generation" of network will have more outnodes
        no_of_out_nodes = len(parent.run(training_set[0])), 
        no_of_hidden_nodes = 100,
        learning_rate = 0.1)

    child2 = NeuralNet(no_of_in_nodes = 28*28,
        #Output vector size is equal to vector size of current network
        #As we create new categories each "generation" of network will have more outnodes
        no_of_out_nodes = len(parent.run(training_set[0])), 
        no_of_hidden_nodes = 100,
        learning_rate = 0.1)

    wrong = 0
    total = 0
    for i in range(0, len(training_set)):
        #Child sees the training image, but the parent network decides what the label should be
        #child network never actually sees the 'real' label (and neither does the parent)

        #Child 1 gets trained by the parent
        label = np.argmax(parent.run(training_set[i]))
        child1.train(training_set[i], oneHot(label, 10))

        #Store how much the parents gets wrong
        if (label != training_label[i]):
            wrong += 1
        total += 1

        #Child 2 gets trained by the onehot labels
        child2.train(training_set[i], training_label_one_hot[i])

    print("Percentage Mislabelled: " + str(wrong/total))
    return child1, child2

def testGenerateChild():
    data = Data()
    parent = NeuralNet(no_of_in_nodes = 28*28, 
        no_of_out_nodes = 10, 
        no_of_hidden_nodes = 100,
        learning_rate = 0.1)

    for i in range(len(data.digits_train_imgs)):
        parent.train(data.digits_train_imgs[i], data.digits_train_labels_one_hot[i])

    #Display Statistics for Parent
    corrects, wrongs = parent.evaluate(data.digits_train_imgs, data.digits_train_labels)
    print("Parent Training Accuracy: ", corrects / ( corrects + wrongs))
    corrects, wrongs = parent.evaluate(data.digits_test_imgs, data.digits_test_labels)
    print("Parent Test Accuracy: ", corrects / ( corrects + wrongs))

    #Generate Children
    child1, child2 = generateChild(parent, data.digits_train_imgs, data.digits_train_labels, data.digits_train_labels_one_hot)
    #child2 = generateChild(parent, data.digits_train_imgs, data.digits_train_labels_one_hot)

    #Display Statistics for Children
    corrects, wrongs = child1.evaluate(data.digits_train_imgs, data.digits_train_labels)
    print("Child1 Training Accuracy: ", corrects / ( corrects + wrongs))
    corrects, wrongs = child1.evaluate(data.digits_test_imgs, data.digits_test_labels)
    print("Child1 Test Accuracy: ", corrects / ( corrects + wrongs))

    corrects, wrongs = child2.evaluate(data.digits_train_imgs, data.digits_train_labels)
    print("Child2 Training Accuracy: ", corrects / ( corrects + wrongs))
    corrects, wrongs = child2.evaluate(data.digits_test_imgs, data.digits_test_labels)
    print("Child2 Test Accuracy: ", corrects / ( corrects + wrongs))

def testOneHot():
    data = Data()
    for i in range(len(data.digits_train_labels)):
        print("Regular: " + str(data.digits_train_labels[i]))
        print("OneHot: " + str(data.digits_train_labels_one_hot[i]))
        print("Generated OneHot: " + str(oneHot(data.digits_train_labels[i], 10)))

def dataSizes():
    data = Data()
    print(len(data.digits_train_imgs))
    print(len(data.digits_test_imgs))
    print(len(data.digits_train_labels))
    print(len(data.digits_test_labels))
    print(len(data.digits_train_labels_one_hot))
    print(len(data.digits_test_labels_one_hot))

    print(len(data.letters_train_imgs))
    print(len(data.letters_test_imgs))
    print(len(data.letters_train_labels))
    print(len(data.letters_test_labels))
    print(len(data.letters_train_labels_one_hot))
    print(len(data.letters_test_labels_one_hot))      

def testSplit():
    dataset = Data().meta_tr()
    for i in range (10):
        print("Splitting into " + str(i) + " pieces")
        print("Returned Length: " + str(len(split(dataset, i))))

def testLabelList():
    no_of_out_nodes = 10
    labelList = np.array([i for i in range(no_of_out_nodes)])
    print(labelList)

def testMeta():
    
    data = Data()
    meta = MetaNet()
    grapher = Grapher()

    #TODO: Actually use this for something
    subNetList = []

    print("Phase 1: Train Subnet on Numbers")
    accuracy = []
    for datum in data.sub_tr():
        (img, label) = datum
        if (meta.trainSubNet(img, label) == label):
            accuracy.append(1)
        else:
            accuracy.append(0)
    grapher.addGraph(accuracy, "SubNet Train Accuracy")
    
    print("Phase 2: Train AlterNet on Letters")
    accuracy = []
    for datum in data.alter_tr():
        (img, label) = datum
        if (meta.trainAlterNet(img, label) == label):
            accuracy.append(1)
        else:
            accuracy.append(0)
    grapher.addGraph(accuracy, "AlterNet Train Accuracy")


    print("Phase 3: Train MetaNet")
    accuracy = []
    for datum in data.meta_tr():
        (img, label, meta_label) = datum
        if (meta.train(img, label, meta_label) == meta_label):
            accuracy.append(1)
        else:
            accuracy.append(0)
    grapher.addGraph(accuracy, "MetaNet Train Accuracy")

    print("Phase 4: Test MetaNet")
    accuracy = []
    for datum in data.meta_te():
        (img, label, meta_label) = datum
        if (meta.run(img) == label):
            accuracy.append(1)
        else:
            accuracy.append(0)
    grapher.addGraph(accuracy, "MetaNet Test Accuracy")
        #Compare meta output with labels / meta_labels

    grapher.graphAll()

#testOneHot()
#testGenerateChild()
#dataSizes()
#testSplit()
#testLabelList()

testMeta()
