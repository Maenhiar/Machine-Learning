import torch
from NetworkFitter.AbstractNetworkTrainer import AbstractNetworkTrainer

class NetworkTrainer(AbstractNetworkTrainer):
    def __init__(self):
        super().__init__()

    def fit(self, dataLoader, model, optimizer, criterion):
        epochLabels = []
        epochPredictions = []
        epochLoss = 0.0
        model.train()
        for input, label in dataLoader:
            optimizer.zero_grad()
            outputs = model(input)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            _, prediction = torch.max(outputs, 1)
            
            epochLabels.extend(label.numpy())
            epochPredictions.extend(prediction.numpy())
            epochLoss += loss.item()
        
        self._updateMetrics(epochLoss / len(dataLoader), epochLabels, epochPredictions)