'''
Functioning or Non-functioning code fragments from various parts of the project
'''


#Convert letter labels to alphanumeric characters
def valueToChar(i):
    letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    return letters[i]

#---------------------------------------
#Originally from NN class
#Statistics generation
#Outsource to tester?
#---------------------------------------

        
#TODO For mixed sets, show which letters / numbers get confused for numbers / letters
def confusion_matrix(self, data_array, labels):
    cm = np.zeros((10, 10), int)
    for i in range(len(data_array)):
        res = self.run(data_array[i])
        res_max = res.argmax()
        target = labels[i][0]
        cm[res_max, int(target)] += 1
    return cm

#This one doesn't work
def meta_confusion_matrix(self, subNN, data, labels, values):
    '''
        Some digits get confused for letters. Some letters get confused for digits
        
        The meta gets labels (0,1) for numbers/letters, but we also need access to to the "real" labels of what is being represented
    '''
    mistakenDigits = np.zeros((10, 26), int)
    mistakenLetters = np.zeros((26, 10), int)
    for i in range(len(data)):
        res = self.run(subNN.run(data[i]))
        res_max = res.argmax()
        target = labels[i][0]
        cm[res_max, int(target)] += 1
    return cm
        
        

def precision(self, label, confusion_matrix):
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label] / col.sum()

def recall(self, label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()
    

#This can't be used for Meta, because the input data is actually pre-processed by another NN first
#Use metaEval instead
def evaluate(self, data, labels):
    corrects, wrongs = 0, 0
    for i in range(len(data)):
        res = self.run(data[i])
        res_max = res.argmax()
        if res_max == labels[i]:
            corrects += 1
        else:
            wrongs += 1
    return corrects, wrongs

def metaEval(self, subNN, data, labels):
    corrects, wrongs = 0, 0
    for i in range(len(data)):
        res = self.run(np.sort(subNN.run(data[i]).T))
        res_max = res.argmax()
        if res_max == labels[i]:
            corrects += 1
        else:
            wrongs += 1
    return corrects, wrongs