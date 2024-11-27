import matplotlib.pyplot as plt

class LossPlotter:
    """
    Thiss class plots the history of training set loss and validation set loss of a model.
    """

    @staticmethod
    def plotLoss(lossHistory):
        """
        This method plots the training set loss and validation set loss chart

        Args:
            lossHistory: the history of the model.
        """
        plt.plot(lossHistory['loss'], label='Loss (train)')
        plt.plot(lossHistory['val_loss'], label='Loss (validation)')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()