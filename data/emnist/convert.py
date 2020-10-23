#Based on this code:
#https://pjreddie.com/projects/mnist-in-csv/

import os

def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

convert("emnist-letters-train-images-idx3-ubyte", "emnist-letters-train-labels-idx1-ubyte",
        "emnist_train.csv", 60000)
convert("emnist-letters-test-images-idx3-ubyte", "emnist-letters-test-labels-idx1-ubyte",
        "emnist_test.csv", 10000)

os.remove("emnist-letters-train-images-idx3-ubyte")
os.remove("emnist-letters-train-labels-idx1-ubyte")
os.remove("emnist-letters-test-images-idx3-ubyte")
os.remove("emnist-letters-test-labels-idx1-ubyte")
