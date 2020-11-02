import pickle
import numpy as np

#Load the data into a dictionary. 
emnist_train_images = []
emnist_train_labels = []
emnist_test_images = []
emnist_test_labels = []

print("loading")
for d in np.genfromtxt('emnist_test.csv', delimiter=','):
    emnist_test_labels.append(d[1])
    emnist_test_images.append(d[1:])

for d in np.genfromtxt('emnist_train.csv', delimiter=','):
    emnist_train_labels.append(d[1])
    emnist_train_images.append(d[1:])

print("saving")
#Pickle the data so we can load it later. 
pickle.dump(emnist_test_images, open("emnist_test_imgs.p", 'wb'))
pickle.dump(emnist_test_labels, open("emnist_test_labels.p", 'wb'))
pickle.dump(emnist_train_images, open("emnist_train_imgs.p", 'wb'))
pickle.dump(emnist_train_labels, open("emnist_train_labels.p", 'wb'))