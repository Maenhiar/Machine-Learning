import matplotlib.pyplot as plt

class ChartPlotter:
    
    @staticmethod
    def plot(epochsNumber, trainingLosses, trainingAccuracies, trainingPrecisions, trainingRecalls, trainingF1Scores):    
        # Creiamo una figura per i grafici
        plt.figure(figsize=(12, 6))

        # Primo grafico per la Loss
        plt.subplot(1, 2, 1)  # Griglia 1x2, posizione 1 (a sinistra)
        plt.plot(range(1, epochsNumber + 1), trainingLosses, label='Loss', color='blue', marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss durante l\'addestramento')
        plt.legend()

        # Secondo grafico per le altre metriche (Accuracy, Precision, Recall, F1 Score)
        plt.subplot(1, 2, 2)  # Griglia 1x2, posizione 2 (a destra)
        plt.plot(range(1, epochsNumber + 1), trainingAccuracies, label='Accuracy', color='green', marker='o')
        plt.plot(range(1, epochsNumber + 1), trainingPrecisions, label='Precision', color='orange', marker='o')
        plt.plot(range(1, epochsNumber + 1), trainingRecalls, label='Recall', color='red', marker='o')
        plt.plot(range(1, epochsNumber + 1), trainingF1Scores, label='F1 Score', color='black', marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Metric Value')
        plt.title('Metriche durante l\'addestramento')
        plt.legend()

        # Mostra il grafico
        plt.tight_layout()
        plt.show()