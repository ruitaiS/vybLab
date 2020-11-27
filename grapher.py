import matplotlib.pyplot as plt


class Graphs:
    def __init__(self):
        self.plot_title="Accuracy Over Time"
        self.window_size=1000
        self.graphList = []

    def addGraph(self, accuracy, name):
        self.graphList.append((accuracy, name))

    def graph(self, graph):
        accuracy, name = graph
        accuracy = [sum(accuracy[i:i+self.window_size])/self.window_size for i in range(len(accuracy)-self.window_size)]
        plt.plot(accuracy, name)

    def graphAll(self):

        for graph in self.graphList:
            graph(graph)

        plt.legend(loc='lower right')
        plt.title(self.plot_title)
        plt.show()

    
