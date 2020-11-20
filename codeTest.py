import numpy as np
from NN import NeuralNet
from data import Data

#convert (integer) label into one-hot format
#Used in metaNN
def oneHot(label, no_categories):
    res = np.full(no_categories,0.01)
    res[int(label)] = 0.99
    return res


def generateChild(parent, training_set, training_label, training_label_one_hot):
    child1 = NeuralNet(no_of_in_nodes = data.image_pixels,
        #Output vector size is equal to vector size of current network
        #As we create new categories each "generation" of network will have more outnodes
        no_of_out_nodes = len(parent.run(training_set[0])), 
        no_of_hidden_nodes = 100,
        learning_rate = 0.1)

    child2 = NeuralNet(no_of_in_nodes = data.image_pixels,
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
    parent = NeuralNet(no_of_in_nodes = data.image_pixels, 
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

def whatsThis():
    data = Data()
    num_train_imgs = data.digits_train_imgs
    num_train_labels = data.digits_train_labels
    char_train_imgs = data.letters_train_imgs
    char_train_labels = data.letters_train_labels

    #Offset char labels so that 
    # Digits: 0-9, characters:10-36
    char_train_labels = [i + 10 for i in char_train_labels]  

    is_char = [1 for i in range(len(char_train_imgs))]
    is_num = [0 for i in range(len(num_train_imgs))]

    #Create a list of tuples (img, label, is_char) for each image
    chars = list(zip(char_train_imgs, np.array(char_train_labels), is_char))

    print(np.array(char_train_labels))


    nums = list(zip(np.array(num_train_imgs)[:,1:], np.array(num_train_labels), is_num))


    combined_data = list(chars + nums)

    #print(combined_data[10])

#testOneHot()
#testGenerateChild()
whatsThis()