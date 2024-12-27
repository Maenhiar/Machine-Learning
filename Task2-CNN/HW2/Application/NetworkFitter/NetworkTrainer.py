import torch
from NetworkFitter.AbstractNetworkTrainer import AbstractNetworkTrainer

class NetworkTrainer(AbstractNetworkTrainer):
    def __init__(self):
        super().__init__()

    def fit(self, dataLoader, model, optimizer, criterion):
        model.train()
        running_loss = 0.0
        labels = []
        predictions = []
        for input, label in dataLoader:
            optimizer.zero_grad()
            outputs = model(input)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, prediction = torch.max(outputs, 1)

            labels.extend(label.numpy())
            predictions.extend(prediction.numpy())

            loss, accuracy, precision, recall, f1Score = \
                self._getMetrics(running_loss, len(dataLoader), labels, predictions, "weighted")

        self._losses.append(loss)
        self._accuracies.append(accuracy)
        self._precisions.append(precision)
        self._recalls.append(recall)
        self._f1Scores.append(f1Score)
