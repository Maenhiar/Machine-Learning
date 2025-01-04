import time
from torch import nn
from DatasetLoader.DatasetLoader import DatasetLoader
from CNNFitting.NetworkTrainer import NetworkTrainer
from CNNFitting.NetworkEvaluator import NetworkEvaluator

class NetworkFitter():
    """
    This class is responsible for managing the training, the evaluation
    and testing of a neural network.

    Attributes:
        __networkTrainer (NetworkTrainer): The object responsible for
                                            training the network.
        __networkValidator (NetworkEvaluator): The object responsible for
                                                performing the evaluation steps.
        __networkTester (NetworkEvaluator): The object responsible for
                                                testing the network.
        __epochsNumber (int): The The number of epochs for which the training will proceed.
        __trainingTime (int): Total training time.
        __trainedModel (nn.Module): The trained model.
    """
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
        """
        Performs model training, validation and testing.

        Args:
        model(nn.Module): The model that must be trained and tested.
        optimizer (optim): The chosen optimizer.
        """
        batchSize = 64
        trainingSetDataLoader, validationSetDataLoader = DatasetLoader().getTrainingSetDataLoader(batchSize)
        criterion = nn.CrossEntropyLoss()
        self.__networkTrainer = NetworkTrainer()
        self.__networkValidator = NetworkEvaluator()

        startTime = time.time()
        
        for i in range(self.__epochsNumber):
            self.__networkTrainer.fit(trainingSetDataLoader, model, optimizer, criterion)
            self.__networkValidator.fit(validationSetDataLoader, model, criterion)

        endTime = time.time()
        self.__trainingTime = endTime - startTime

        testSetDataLoader = DatasetLoader().getTestSetDataLoader(batchSize)
        self.__networkTester = NetworkEvaluator()
        self.__networkTester.fit(testSetDataLoader, model, criterion)
        self.__trainedModel = model