import torch
from NetworkFitter.AbstractNetworkTrainer import AbstractNetworkTrainer

class NetworkEvaluator(AbstractNetworkTrainer):
    __predictions = []
    __labels = []

    def __init__(self):
        super().__init__()

    def fit(self, dataLoader, model, criterion):
        self.__labels = []
        self.__predictions = []
        running_loss = 0.0
        model.eval()
        with torch.no_grad():
            for input, label in dataLoader:
                outputs = model(input)
                _, prediction = torch.max(outputs, 1)

                loss = criterion(outputs, label)
                running_loss += loss.item()
                _, prediction = torch.max(outputs, 1)

                self.__labels.extend(label.numpy())
                self.__predictions.extend(prediction.numpy())

                loss, accuracy, precision, recall, f1Score = \
                    self._getMetrics(running_loss, len(dataLoader), self.__labels, self.__predictions, "weighted")

            self._losses.append(loss)
            self._accuracies.append(accuracy)
            self._precisions.append(precision)
            self._recalls.append(recall)
            self._f1Scores.append(f1Score)

    def getPredictions(self):
        return self.__predictions.copy()

    def getLabels(self):
        return self.__labels.copy()
