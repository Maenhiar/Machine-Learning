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

        # Plot dell'errore assoluto medio (MAE)
        plt.plot(lossHistory['mean_squared_error'], label='MAE (train)')
        plt.plot(lossHistory['val_mean_squared_error'], label='MAE (validation)')
        plt.title('Training and Validation MAE')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Absolute Error')
        plt.legend()
        plt.show()