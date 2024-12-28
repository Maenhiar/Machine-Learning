import time
from torch import nn
from DatasetLoader.DatasetLoader import DatasetLoader
from NetworkFitter.NetworkTrainer import NetworkTrainer
from NetworkFitter.NetworkEvaluator import NetworkEvaluator

class NetworkFitter():
    __networkTrainer = NetworkTrainer()
    __networkValidator = NetworkEvaluator()
    __networkTester = NetworkEvaluator()
    __epochsNumber = 50
    __trainingTime = 0
    __trainedModel = None

    def __init__(self):
        pass

    def getTrainedModel(self):
        return self.__trainedModel
    
    def getEpochsNumber(self):
        return self.__epochsNumber

    def getTrainingTime(self):
        return self.__trainingTime
    
    def getTrainingMetrics(self):
        return self.__networkTrainer.getAllMetrics()
    
    def getValidationMetrics(self):
        return self.__networkValidator.getAllMetrics()
    
    def getTestMetrics(self):
        return self.__networkTester.getAllMetrics()
    
    def fit(self, model, optimizer):
        batchSize = 64
        trainingSetDataLoader, validationSetDataLoader = DatasetLoader.getTrainingSetDataLoader(batchSize)
        criterion = nn.CrossEntropyLoss()
        self.__networkTrainer = NetworkTrainer()
        self.__networkValidator = NetworkEvaluator()

        startTime = time.time()
        # Fit the model with training set and validation set
        for _ in range(self.__epochsNumber):
            self.__networkTrainer.fit(trainingSetDataLoader, model, optimizer, criterion)
            self.__networkValidator.fit(validationSetDataLoader, model, criterion)

        endTime = time.time()
        self.__trainingTime = endTime - startTime

        # Testing the trained model with test set
        testSetDataLoader = DatasetLoader.getTestSetDataLoader(batchSize)
        self.__networkTester = NetworkEvaluator()
        self.__trainedModel = self.__networkTester.fit(testSetDataLoader, model, criterion)