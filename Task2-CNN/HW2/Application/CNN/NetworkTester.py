import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class NetworkTester:
            
    @staticmethod
    def test_loop(testSetDataLoader, model):
        labels = []
        predictions = []
        
        model.eval()
        with torch.no_grad():
            for input, label in testSetDataLoader:
                outputs = model(input)
                _, prediction = torch.max(outputs, 1)
                labels.extend(label.numpy())
                predictions.extend(prediction.numpy())
        
            # Calcolo delle metriche globali (weighted average)
            metricAverage = "weighted"
            accuracy = accuracy_score(labels, predictions)
            precision = precision_score(labels, predictions, average = metricAverage)
            recall = recall_score(labels, predictions, average = metricAverage)
            f1Score = f1_score(labels, predictions, average = metricAverage)
            
            # Calcolo delle metriche per ogni classe separatamente
            taskLabels = [0, 1, 2, 3, 4]
            classesPrecisions = precision_score(labels, predictions, average = None, labels = taskLabels.copy())
            classesRecalls = recall_score(labels, predictions, average = None, labels = taskLabels.copy())
            classesF1Scores = f1_score(labels, predictions, average = None, labels = taskLabels.copy())

            return accuracy, precision, recall, f1Score, classesPrecisions, classesRecalls, \
                classesF1Scores, labels, predictions
