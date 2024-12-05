import time
import numpy as np
from sklearn.model_selection import KFold

class KFoldCrossValidation():
    """
    This static class performs the K-Fold crossvalidation of the model
    passed as input.
    """
    @staticmethod
    def performKFoldCrossValidation(modelsList: list, trainingSetInput, trainsetOuput, testSetInput, testSetOutput):
        """
        Performs the K-Fold crossvalidation of the model.
        The dataset is splitted in training set (2/3 of the dataset)
        and validation set (1/3 of the dataset).
        
        Args:
            modelsList: a list of 3 models (one for each crossvalidation fold).
                        It is necessary due to the impossibility to clone a keras
                        model without having any error or side effect.
        
        Returns:
            bestModel: an object containing the model that best performed 
                        during the crossvalidation along with its performances parameters.
        """
        if len(modelsList) != 3:
            raise ValueError("The list must contain three models of the same type.")
        
        mseScores = []
        testSetMSEScores = []
        modelHistories = []

        kFoldCrossValidation = KFold(n_splits = 3, shuffle = True)
        
        i = 0
        startTime = time.time()
        for train_index, val_index in kFoldCrossValidation.split(trainingSetInput):
            trainingFoldInput = trainingSetInput[train_index]
            validationFoldInput = trainingSetInput[val_index]
            trainingFoldOutput = trainsetOuput[train_index]
            validationFoldOutput = trainsetOuput[val_index]

            history = modelsList[i]._getModel().fit(
                trainingFoldInput,
                trainingFoldOutput,
                batch_size = modelsList[i].getBatchSize(),
                epochs = modelsList[i].getEpochsNumber(),
                callbacks = modelsList[i].getEarlyStopping(),
                validation_data = (validationFoldInput, validationFoldOutput)
            )

            modelHistories.append(history)

            mse = modelsList[i]._getModel().evaluate(validationFoldInput, validationFoldOutput)
            mseScores.append(mse)

            testSetMSE = modelsList[i]._getModel().evaluate(testSetInput, testSetOutput)
            testSetMSEScores.append(testSetMSE)

            i = i + 1
        
        endTime = time.time()

        finalMSE = np.mean(mseScores)
        finalTestSetMSE = np.mean(testSetMSEScores)

        modelPerformances = {   
            "avgMSE": finalMSE,
            "avgTestSetMSE": finalTestSetMSE,
            "trainingTime": endTime - startTime,
            "histories" : modelHistories
        } 
        
        return modelPerformances