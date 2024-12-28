import torch
from NetworkFitter.AbstractNetworkTrainer import AbstractNetworkTrainer

class NetworkEvaluator(AbstractNetworkTrainer):
    def __init__(self):
        super().__init__()

    def fit(self, dataLoader, model, criterion):
        epochLabels = []
        epochPredictions = []
        epochLoss = 0.0
        model.eval()
        with torch.no_grad():
            for input, label in dataLoader:
                outputs = model(input)
                loss = criterion(outputs, label)

                _, prediction = torch.max(outputs, 1)

                epochLabels.extend(label.numpy())
                epochPredictions.extend(prediction.numpy())
                epochLoss += loss.item()

            self._updateMetrics(epochLoss / len(dataLoader), epochLabels, epochPredictions)
    
        return model