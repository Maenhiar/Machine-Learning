import time
import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf

class KFoldCrossValidation():

    def performKFoldCrossValidation(self, modelsList: list):
        if len(modelsList) != 3:
            raise ValueError("La lista deve contenere esattamente 3 modelli.")
        
        mseScores = []
        testSetMSEScores = []
        modelsHistory = []
                                     
        trainingSetInput = modelsList[0].getTraingSetInput()
        trainsetOuput = modelsList[0].getTraingSetOutput()
        testSetInput = modelsList[0].getTestSetInput()
        testSetOutput = modelsList[0].getTestSetOutput()

        kFoldCrossValidation = KFold(n_splits = 3, shuffle = True)

        weights = modelsList[0].getModel().get_weights()

        i = 0
        startTime = time.time()
        for train_index, val_index in kFoldCrossValidation.split(trainingSetInput):
            trainingFoldInput = trainingSetInput[train_index]
            validationFoldInput = trainingSetInput[val_index]
            trainingFoldOutput = trainsetOuput[train_index]
            validationFoldOutput = trainsetOuput[val_index]

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

        bestModel = self.BestModelPerformances(finalMSE, finalTestingSetMSE, 
                                                modelsHistory[bestPerformanceModelIndex], 
                                                    endTime - startTime,
                                                        modelsList[bestPerformanceModelIndex].getModel())
        
        return bestModel     

    class BestModelPerformances():
        __finalMSE = None
        __finalTestSetMSE = None
        __modelHistory = None
        __trainingTime = 0
        __model = None

        def __init__(self, finalMSE, finalTestSetMSE, modelHistory, trainingTime, model):
            self.__finalMSE = finalMSE
            self.__finalTestSetMSE = finalTestSetMSE
            self.__modelHistory = modelHistory
            self.__trainingTime = trainingTime
            self.__model = model
        
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

