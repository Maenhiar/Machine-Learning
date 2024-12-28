from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class AbstractNetworkTrainer(ABC):

    def __init__(self):
        self._losses = []
        self._accuracies = []
        self._precisions = []
        self._recalls = []
        self._f1Scores = []
        self._labels = []
        self._predictions = []

    def getAllMetrics(self):
        return self._losses.copy(), self._accuracies.copy(), self._precisions.copy(), self._recalls.copy(), \
                self._f1Scores.copy(), self._labels.copy(), self._predictions.copy()

    
    @abstractmethod
    def fit(self, dataLoader, model, optimizer, criterion):
        pass

    def _updateMetrics(self, loss, labels, predictions):        
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

        self._labels.clear()
        self._labels.extend(labels)

        self._predictions.clear()
        self._predictions.extend(predictions)