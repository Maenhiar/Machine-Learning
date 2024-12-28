import matplotlib.pyplot as plt

class ChartPlotter:
    """
    This static class allows to plot the provided
    performance metric into a chart.
    """
    @staticmethod
    def plot(epochsNumber, metricName, trainingMetric, validationMetric):
        """
        Plots the provided performance metric into a chart.

        Args:
        epochsNumber(int): The total number of epochs for which the network has been trained.
        metricName(string): The provided metric name.
        trainingMetric(list): The list of the training metric values.
        validationMetric(list): The list of the validation metric values.
        """
        plt.plot(range(1, epochsNumber + 1), trainingMetric, color = "blue", label = f"Training - {metricName}")
        plt.plot(range(1, epochsNumber + 1), validationMetric, color = "orange", label = f"Validation - {metricName}")
        plt.xlabel("Epochs")
        plt.ylabel(metricName)  
        plt.title(f"Training and Validation - {metricName}")
        plt.legend()
        plt.show()