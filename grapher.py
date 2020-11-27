import matplotlib.pyplot as plt


class Grapher:
    def __init__(self):
        self.plot_title="Accuracy Over Time"
        self.window_size=1000
        self.graphList = []

    def addGraph(self, accuracy, name):
        self.graphList.append((accuracy, name))

    def graph(self, graph):
        accuracy, name = graph
        accuracy = [sum(accuracy[i:i+self.window_size])/self.window_size for i in range(len(accuracy)-self.window_size)]
        plt.plot(accuracy, label=name)

    def graphAll(self):

        for element in self.graphList:
            self.graph(element)

        plt.legend(loc='lower right')
        plt.title(self.plot_title)
        plt.show()

    
