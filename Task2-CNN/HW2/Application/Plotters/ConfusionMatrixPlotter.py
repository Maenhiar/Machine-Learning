import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class ConfusionMatrixPlotter:
    """
    This static class allows to plot the 
    confusion matrix.
    """
    @staticmethod
    def plot(labels, predictions):
        """
        Plots the confusion matrix.

        Args:
        labels(list): the dataset labels.
        predictions(list): the network predictions.
        """
        cm = confusion_matrix(labels, predictions)

        plt.imshow(cm, cmap = "Blues", interpolation = "nearest")
        plt.title("Confusion Matrix")
        plt.colorbar()
        
        plt.xticks(np.arange(len(cm)), np.arange(len(cm)))
        plt.yticks(np.arange(len(cm)), np.arange(len(cm)))
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha = "center", va = "center", color = "white" if cm[i, j] > cm.max() / 2 else "black")
        
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.show()