import matplotlib.pyplot as plt

def graph(accuracy, meta_accuracy):
    print("Reporting Results")
    
    plot_title="Accuracy Over Time"
    window_size=1000

    meta_accuracy = [sum(meta_accuracy[i:i+window_size])/window_size for i in range(len(meta_accuracy)-window_size)]
    accuracy = [sum(accuracy[i:i+window_size])/window_size for i in range(len(accuracy)-window_size)]

    plt.plot(accuracy, label="Model Accuracy")
    plt.plot(meta_accuracy, label="Meta Accuracy")
    plt.legend(loc='lower right')
    plt.title(plot_title)
    plt.show()