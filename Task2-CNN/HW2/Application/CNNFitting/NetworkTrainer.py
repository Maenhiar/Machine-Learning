import torch
from CNNFitting.AbstractClassificationNN import AbstractClassificationNN

class NetworkTrainer(AbstractClassificationNN):
    """
    This class performs the training of a 
    classification neural network.
    """
    def __init__(self):
        super().__init__()

    def fit(self, dataLoader, model, optimizer, criterion):
        """
        Trains the model with the provided hyperparameters.

        Args:
        dataLoader (DataLoader): The training set.
        model(nn.Module): The model that must be trained.
        optimizer (optim): The chosen optimizer.
        criterion (nn): The chosen loss function.
        """
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