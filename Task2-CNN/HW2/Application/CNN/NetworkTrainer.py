import time
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class NetworkTrainer:
    
    @staticmethod
    def train_loop(trainingSetDataLoader, model, criterion, optimizer, epochs):
        trainingLosses = []
        trainingAccuracies = []
        trainingPrecisions = []
        trainingRecalls = []
        trainingF1Scores = []
        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        model.train()
        startingTime = time.time()
        for epoch in range(epochs):
            running_loss = 0.0
            labels = []
            predictions = []
            for input, label in trainingSetDataLoader:
                optimizer.zero_grad()
                outputs = model(input)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()

                # statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)  # Trova l'indice della classe con probabilità massima
                labels.extend(label.numpy())
                predictions.extend(predicted.numpy())

            # Calcoliamo la media della loss per questa epoca
            epochLoss = running_loss / len(trainingSetDataLoader)
            trainingLosses.append(epochLoss)

            # Calcoliamo le metriche
            accuracy = accuracy_score(labels, predictions)
            metricAverage = "weighted"
            precision = precision_score(labels, predictions, average = metricAverage, zero_division = 1)
            recall = recall_score(labels, predictions, average = metricAverage, zero_division = 1)
            f1Score = f1_score(labels, predictions, average = metricAverage)

            trainingAccuracies.append(accuracy)
            trainingPrecisions.append(precision)
            trainingRecalls.append(recall)
            trainingF1Scores.append(f1Score)

            print(f"Epoch [{epoch + 1} / {epochs}], Loss: {epochLoss:.4f}, Accuracy: {accuracy:.4f}, \
                  Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1Score:.4f}")

        endingTime = time.time()
        trainingTime = endingTime - startingTime

        return trainingLosses, trainingAccuracies, trainingPrecisions, trainingRecalls, trainingF1Scores, trainingTime