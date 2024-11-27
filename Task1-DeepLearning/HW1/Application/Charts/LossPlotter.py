import matplotlib.pyplot as plt

class LossPlotter:

    @staticmethod
    def plotLoss(lossHistory):
        # Plot della loss
        plt.plot(lossHistory['loss'], label='Loss (train)')
        plt.plot(lossHistory['val_loss'], label='Loss (validation)')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()