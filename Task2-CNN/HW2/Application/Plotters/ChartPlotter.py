import matplotlib.pyplot as plt

class ChartPlotter:
    
    @staticmethod
    def plot(epochsNumber, metricName, trainingMetric, validationMetric):    
        plt.plot(range(1, epochsNumber + 1), trainingMetric, color = "blue", label = f"Training - {metricName}")
        plt.plot(range(1, epochsNumber + 1), validationMetric, color = "orange", label = f"Validation - {metricName}")
        plt.xlabel("Epochs")
        plt.ylabel(metricName)  
        plt.title(f"Training and Validation - {metricName}")
        plt.legend()
        plt.show()