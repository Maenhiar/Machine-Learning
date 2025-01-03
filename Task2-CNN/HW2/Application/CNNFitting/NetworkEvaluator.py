import torch
from CNNFitting.AbstractClassificationNN import AbstractClassificationNN

class NetworkEvaluator(AbstractClassificationNN):
    """
    Depending on the context, this class can be used either 
    to perform the validation steps or for testing a classification neural network.
    """
    def __init__(self):
        super().__init__()

    def fit(self, dataLoader, model, criterion):
        """
        Trains the model with the provided hyperparameters.

        Args:
        dataLoader (DataLoader): The validation set or the test set, depending on the context.
        model (nn.Module): The model that must be validated or tested, depending on the context.
        criterion (nn): The chosen loss function.
        """
        labels = []
        predictions = []
        loss = 0.0
        model.eval()
        with torch.no_grad():
            for input, label in dataLoader:
                outputs = model(input)
                currentLoss = criterion(outputs, label)

                _, prediction = torch.max(outputs, 1)

                labels.extend(label.numpy())
                predictions.extend(prediction.numpy())
                loss += currentLoss.item()

            self._updateMetrics(loss / len(dataLoader), labels, predictions)