import time
import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
from Jacobian.JacobiansComparator import FK_Jacobian, Jacobian_analytical

class KFoldCrossValidation():
    __mseScores = []  # Lista per raccogliere gli MSE per ogni fold
    __finalMSE = [] # Inizializza il test set per la valutazione finale

    def performKFoldCrossValidation(self, modelsList: list):
        if len(modelsList) != 3:
            raise ValueError("La lista deve contenere esattamente 3 modelli.")
                             
        trainingSetInput = modelsList[0].getTraingSetInput()
        trainsetOuput = modelsList[0].getTraingSetOutput()
        testSetInput = modelsList[0].getTestSetInput()
        testSetOutput = modelsList[0].getTestSetOutput()

        kFoldCrossValidation = KFold(n_splits = 3)

        weights = modelsList[0].getModel().get_weights()

        i = 0
        for train_index, val_index in kFoldCrossValidation.split(trainingSetInput):
            trainingFoldInput = trainingSetInput[train_index]
            validationFoldInput = trainingSetInput[val_index]
            trainingFoldOutput = trainsetOuput[train_index]
            validationFoldOutput = trainsetOuput[val_index]

            start_time = time.time()

            modelsList[i].getModel().set_weights(weights)  
            
            history = modelsList[i].getModel().fit(
                trainingFoldInput,
                trainingFoldOutput,
                batch_size = modelsList[i].getBatchSize(),
                epochs = modelsList[i].getEpochsNumber(),
                callbacks = modelsList[i].getEarlyStopping(),
                validation_data=(validationFoldInput, validationFoldOutput)
            )

            end_time = time.time()

            # Calcola e stampa la durata dell'allenamento
            training_time = end_time - start_time
            print(f'Tempo di allenamento: {training_time:.4f} secondi')

            print(history.history)

            mse_val = modelsList[i].getModel().evaluate(validationFoldInput, validationFoldOutput, verbose=0)
            self.__mseScores.append(mse_val)

            self.__finalMSE.append(modelsList[i].getModel().evaluate(testSetInput, testSetOutput))

            i = i + 1

        # Calcola la media dell'MSE sui fold di validazione
        print(f"Mean Validation MSE across all folds: {np.mean(self.__mseScores)}")

        # Calcola la valutazione finale del modello sui test set
        print(f"Final MSE on Test Set: {np.mean(self.__finalMSE)}")

        bestPerformanceModelIndex = self.__finalMSE.index(min(self.__finalMSE))
        model = modelsList[bestPerformanceModelIndex].getModel()
        
        # Calcolare la posizione finale per theta
        theta = tf.constant([0.5, 1.0], dtype=tf.float32)  # Angoli di esempio

        # Jacobiano computato tramite il modello
        jacobian_computed = FK_Jacobian(model, theta)

        # Jacobiano analitico
        theta_numpy = theta.numpy()  # Converti theta in formato numpy per il calcolo analitico
        jacobian_analytical = Jacobian_analytical(theta_numpy)

        # Confronta i Jacobiani
        print("Jacobiano computato:\n", jacobian_computed.numpy())
        print("Jacobiano analitico:\n", jacobian_analytical)
        print("Differenza tra i Jacobiani:", np.abs(jacobian_computed.numpy() - jacobian_analytical))

