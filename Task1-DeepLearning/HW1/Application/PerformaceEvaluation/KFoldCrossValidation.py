import time
import numpy as np
from sklearn.model_selection import KFold

class KFoldCrossValidation():
    """
    This class performs the K-Fold crossvalidation of the model
    passed as input.
    """

    def performKFoldCrossValidation(self, modelsList: list, trainingSetInput, trainsetOuput, testSetInput, testSetOutput):
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
        modelsHistory = []
        weights = modelsList[0].getModel().get_weights()
        kFoldCrossValidation = KFold(n_splits = 3, shuffle = True)
        
        i = 0
        startTime = time.time()
        for train_index, val_index in kFoldCrossValidation.split(trainingSetInput):
            trainingFoldInput = trainingSetInput[train_index]
            validationFoldInput = trainingSetInput[val_index]
            trainingFoldOutput = trainsetOuput[train_index]
            validationFoldOutput = trainsetOuput[val_index]

            """Necessary if we want that each model is initialized 
                with the same weights without incurring in any error."""
            modelsList[i].getModel().set_weights(weights)  
            
            history = modelsList[i].getModel().fit(
                trainingFoldInput,
                trainingFoldOutput,
                batch_size = modelsList[i].getBatchSize(),
                epochs = modelsList[i].getEpochsNumber(),
                callbacks = modelsList[i].getEarlyStopping(),
                validation_data=(validationFoldInput, validationFoldOutput)
            )

            modelsHistory.append(history.history)

            mse = modelsList[i].getModel().evaluate(validationFoldInput, validationFoldOutput)
            mseScores.append(mse)

            testSetMSE = modelsList[i].getModel().evaluate(testSetInput, testSetOutput)
            testSetMSEScores.append(testSetMSE)

            i = i + 1
        
        endTime = time.time()

        finalMSE = np.mean(mseScores)
        finalTestingSetMSE = np.mean(testSetMSEScores)

        bestPerformanceModelIndex = testSetMSEScores.index(min(testSetMSEScores))

        bestModel = self.BestModelPerformances(finalMSE, finalTestingSetMSE, modelsHistory[bestPerformanceModelIndex], 
                                                    endTime - startTime, modelsList[bestPerformanceModelIndex].getModel(),
                                                        modelsList[bestPerformanceModelIndex].getEarlyStopping())
        
        return bestModel     

    class BestModelPerformances():
        """
        This class represents a model along with its performance parameters.
        """
        __finalMSE = None
        __finalTestSetMSE = None
        __modelHistory = None
        __trainingTime = 0
        __model = None
        __earlyStopping = None

        def __init__(self, finalMSE, finalTestSetMSE, modelHistory, trainingTime, model, earlyStopping):
            self.__finalMSE = finalMSE
            self.__finalTestSetMSE = finalTestSetMSE
            self.__modelHistory = modelHistory
            self.__trainingTime = trainingTime
            self.__model = model
            self.__earlyStopping = earlyStopping
        
        def getFinalMSE(self):
            return self.__finalMSE
        
        def getFinalTestSetMSE(self):
            return self.__finalTestSetMSE
        
        def getModelHistory(self):
            return self.__modelHistory
        
        def getTrainingTime(self):
            return self.__trainingTime
        
        def getModel(self):
            return self.__model
        
        def getEarlyStopping(self):
            return self.__earlyStopping

