from sklearn.datasets import load_iris
iris = load_iris()
import numpy as np
import matplotlib.pyplot as plt



#print(iris.data.shape)

#Iris is a Bunch object (similar to dict)
#print(type(iris))

n_samples, n_features = iris.data.shape
#print("Number of Samples: ", n_samples)
#print("Number of features: ", n_features)

#The sepal length, sepal width, petal length, and petal width of the first flower
#print(iris.data[0])

#print(len(iris.target))
#print(iris.target)

#print(iris.target.shape)


#Numpy bincount can list the class distribution
#Bincount(input) creates an array where output[i] = the number of times i comes up in input
#print(np.bincount([0, 5]))
#print(np.bincount(iris.target))

#Names for each class are stored in target_names
#print(iris.target_names)


#-------------Visualizing features

#Not totally sure about the syntax used to select from the arrays below
#I might be wrong, but it's my best guess afaik

#Prints the first five elements of iris.data where the target == 1
#print(iris.data[iris.target == 1][:5])

#Prints an array containing only the first element of each of the arrays from the above
#print(iris.data[iris.target == 1, 0][:5])


#IDK what all this is but it shows some cool stuff:
fig, ax = plt.subplots()
x_index = 3
colors = ['blue', 'red', 'green']

for label, color in zip(range(len(iris.target_names)), colors):
    ax.hist(iris.data[iris.target==label, x_index], 
            label=iris.target_names[label],
            color=color)

ax.set_xlabel(iris.feature_names[x_index])
ax.legend(loc='upper right')
fig.show()

