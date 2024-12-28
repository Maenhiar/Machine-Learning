from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class AbstractClassificationNN(ABC):
    """
    This abstract class defines the common methods used for 
    training, validating, and testing a classification neural network. It also
    stores the performances metrics.
    """
    def __init__(self):
        self._losses = []
        self._accuracies = []
        self._precisions = []
        self._recalls = []
        self._f1Scores = []
        self._labels = []
        self._predictions = []

    def getAllMetrics(self):
        """
        This method returns all classification performance metrics
        along with predictions and labels.
        """
        return self._losses.copy(), self._accuracies.copy(), self._precisions.copy(), self._recalls.copy(), \
                self._f1Scores.copy(), self._labels.copy(), self._predictions.copy()
    
    @abstractmethod
    def fit(self, dataLoader, model, optimizer, criterion):
        """
        This abstract method will be implemented in order to
        train, validate or test the neural network depending on the context.
        """
        pass

    def _updateMetrics(self, loss, labels, predictions):
        # This method updates the classification metrics of
        # the network along with predictions and labels.
        
        self._losses.append(loss)

        accuracy = accuracy_score(labels, predictions)
        self._accuracies.append(accuracy)

        meanMetric = "weighted"
        precision = precision_score(labels, predictions, average = meanMetric, zero_division = 1)
        self._precisions.append(precision)

        recall = recall_score(labels, predictions, average = meanMetric, zero_division = 1)
        self._recalls.append(recall)

        f1Score = f1_score(labels, predictions, average = meanMetric)
        self._f1Scores.append(f1Score)

        # Replace labels and predictions lists content because
        # only last epochs values must be taken.
        self._labels.clear()
        self._labels.extend(labels)
        self._predictions.clear()
        self._predictions.extend(predictions)