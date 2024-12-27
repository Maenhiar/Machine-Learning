from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class AbstractNetworkTrainer(ABC):

    def __init__(self):
        self._losses = []
        self._accuracies = []
        self._precisions = []
        self._recalls = []
        self._f1Scores = []

    def getLosses(self):
        return self._losses.copy()

    def getAccuracies(self):
        return self._accuracies.copy()

    def getPrecisions(self):
        return self._precisions.copy()

    def getRecalls(self):
        return self._recalls.copy()

    def getF1Scores(self):
        return self._f1Scores.copy()
    
    @abstractmethod
    def fit(self, dataLoader, model, optimizer, criterion):
        pass

    def _getMetrics(self, running_loss, datasetLength, labels, predictions, meanMetric):
        loss = running_loss / datasetLength
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average = meanMetric, zero_division = 1)
        recall = recall_score(labels, predictions, average = meanMetric, zero_division = 1)
        f1Score = f1_score(labels, predictions, average = meanMetric)
        return loss, accuracy, precision, recall, f1Score