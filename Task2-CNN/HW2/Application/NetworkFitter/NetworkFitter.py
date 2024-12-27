import time
from torch import nn
import torch.optim as optim
from DatasetLoader.DatasetLoader import DatasetLoader
from CNN.CarRacingCNN import CarRacingCNN
from NetworkFitter.NetworkTrainer import NetworkTrainer
from NetworkFitter.NetworkEvaluator import NetworkEvaluator

class NetworkFitter():
    __model = None
    __networkTrainer = NetworkTrainer()
    __networkValidator = NetworkEvaluator()
    __networkTester = NetworkEvaluator()
    __epochsNumber = 50
    __trainingTime = 0

    def __init__(self):
        self.__fit()

    def getEpochsNumber(self):
        return self.__model.copy()
    
    def getEpochsNumber(self):
        return self.__epochsNumber

    def getTrainingTime(self):
        return self.__trainingTime
    
    def getTrainingMetrics(self):
        return self.__getMetrics(self.__networkTrainer)
    
    def getValidationMetrics(self):
        return self.__getMetrics(self.__networkValidator)
    
    def getTestMetrics(self):
        return self.__getMetrics(self.__networkTester), self.__networkTester.getPredictions(), \
                self.__networkTester.getLabels()

    def __getMetrics(self, networkFitter):
        return networkFitter.getLosses(), networkFitter.getAccuracies(), \
            networkFitter.getPrecisions(), networkFitter.getRecalls(), networkFitter.getF1Scores()
    
    def __fit(self):
        batchSize = 64
        trainingSetDataLoader, validationSetDataLoader = DatasetLoader.getTrainingSetDataLoader(batchSize)
        self.__model = CarRacingCNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.__model.parameters(), lr = 0.001, momentum = 0.9)
        self.__networkTrainer = NetworkTrainer()
        self.__networkValidator = NetworkEvaluator()

        startTime = time.time()
        for _ in range(self.__epochsNumber):
            self.__networkTrainer.fit(trainingSetDataLoader, self.__model, optimizer, criterion)
            self.__networkValidator.fit(validationSetDataLoader, self.__model, criterion)

        endTime = time.time()
        self.__trainingTime = endTime - startTime

        testSetDataLoader = DatasetLoader.getTestSetDataLoader(batchSize)
        self.__networkTester = NetworkEvaluator()
        self.__networkTester.fit(testSetDataLoader, self.__model, criterion)

