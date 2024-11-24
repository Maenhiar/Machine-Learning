from copy import deepcopy
import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf

class KFoldCrossValidation():
    __trainsetInput = None
    __trainsetOuput = None
    __testestInput = None
    __testsetOutput = None
    __kFoldCrossValidation = None
    __mse_scores = []  # Lista per raccogliere gli MSE per ogni fold
    __final_mse = [] # Inizializza il test set per la valutazione finale

    def __init__(self, trainsetInput, trainsetOuput, testestInput, testsetOutput):
        self.__trainsetInput = trainsetInput
        self.__trainsetOuput = trainsetOuput
        self.__testestInput = testestInput
        self.__testsetOutput = testsetOutput
        self.__kFoldCrossValidation = KFold(n_splits = 5, shuffle = True, random_state = 42)  # 5 fold

    def performKFoldCrossValidation(self, model, batchSize, epochsNunmber, callbacks):
        for train_index, val_index in self.__kFoldCrossValidation.split(self.__trainsetInput):
            # Dividi il dataset in training e validation set
            X_train, X_val = self.__trainsetInput[train_index], self.__trainsetInput[val_index]
            y_train, y_val = self.__trainsetOuput[train_index], self.__trainsetOuput[val_index]

            clonedModel = deepcopy(model)

            history = clonedModel.fit(
                X_train,
                y_train,
                batch_size = batchSize,
                epochs = epochsNunmber,
                callbacks = callbacks,
                validation_data=(X_val, y_val)
            )
            

        print(history.history)
        
        # Calcola l'errore quadratico medio (MSE) per il fold di validazione
        mse_val = clonedModel.evaluate(X_val, y_val, verbose=0)
        self.__mse_scores.append(mse_val)
        
        # Alla fine, valuta anche il modello sul test set separato
        self.__final_mse.append(clonedModel.evaluate(self.__testestInput, self.__testsetOutput))

        # Calcola la media dell'MSE sui fold di validazione
        print(f"Mean Validation MSE across all folds: {np.mean(self.__mse_scores)}")

        # Calcola la valutazione finale del modello sui test set
        print(f"Final MSE on Test Set: {np.mean(self.__final_mse)}")