from abc import ABC
import tensorflow as tf
import numpy as np
import random
import os
import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers

class NeuralNetwork(ABC):
    """
    This abstract class provides the common features to instantiate a Keras 
    feedforward neural network model for a forward kinematics problem, ensuring 
    no errors in initialization or components addition. It provides methods that 
    allow for easily adding and retrieving the neural network's components, such as 
    layers, neurons, dropout, epochs, batch size, dropout, L1 or L2 regularization and early stopping settings.
    This class also ensures that hidden layers activation function is always ReLU and 
    output layer activation function is always linear because this is the 
    most common efficient choice in regression problems tackled with a feedforward neural networ.
    The chosen loss function is Mean Squared Error and the optimizer is Adam with a configurable learning rate.
    """
    __inputSize = 2
    __outputSize = 2
    __batchSize = 32
    __epochsNumber = 10
    __model = None
    __isEarlyStoppingActivated = False
    __earlyStoppingCallback = None
    __activationFunctionsButLast = "relu"
    __nnLoss = "mean_squared_error"
    
    def __init__(self, useStaticSeed: bool):
        if type(self) is NeuralNetwork:
            raise NotImplementedError("Cannot instance an abstract class.")

        if useStaticSeed :

            # Setting the seed
            seed_value = 8

            # Set the seed in Python and in other libraries
            random.seed(seed_value)
            np.random.seed(seed_value)

            # Set the seed in TensorFlow
            tf.random.set_seed(seed_value)

            # Set the seed using keras.utils.set_random_seed. This will set:
            # 1) `numpy` seed
            # 2) backend random seed
            # 3) `python` random seed
            keras.utils.set_random_seed(seed_value)

            # If using TensorFlow, this will make GPU ops as deterministic as possible,
            # but it will affect the overall performance, so be mindful of that.
            tf.config.experimental.enable_op_determinism()

            # Configure the TensorFlow session for deterministic behavior
            os.environ['PYTHONHASHSEED'] = str(seed_value)

            # Set the configuration for TensorFlow
            physical_devices = tf.config.list_physical_devices('GPU')
            if len(physical_devices) > 0:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                tf.config.set_logical_device_configuration(physical_devices[0], 
                    [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])

            # Set options for deterministic operations in TensorFlow
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        
        self.__model = Sequential()
        self.__model.add(Dense(units = self.__inputSize, input_dim = self.__inputSize, activation = self.__activationFunctionsButLast))
        
    def getEarlyStopping(self):
        return self.__earlyStoppingCallback if self.__isEarlyStoppingActivated else []
    
    def getBatchSize(self):
        return self.__batchSize
    
    def getEpochsNumber(self):
        return self.__epochsNumber
    
    def getModel(self):
        return self.__model
    
    def addDenseLayer(self, neuronsNumber: int):
        if self.__isNeuronsNumberValid(neuronsNumber) == False :
            raise ValueError("Neurons number cannot be a negative value")
        
        self.__model.add(Dense(neuronsNumber, activation = self.__activationFunctionsButLast))

    def addDenseLayerWithL1Regularization(self, neuronsNumber: int, penaltyValue = float):
        if self.__isNeuronsNumberValid(neuronsNumber) == False or \
            self.__isLRegularizationPenaltyValueValid(penaltyValue) == False :
                raise ValueError("Neurons number and regularization penalty cannot be a negative value")
        
        self.__model.add(Dense(neuronsNumber, activation = self.__activationFunctionsButLast, 
                                   kernel_regularizer = regularizers.l1(penaltyValue)))

    def addDenseLayerWithL2Regularization(self, neuronsNumber: int, penaltyValue = float):
        if self.__isNeuronsNumberValid(neuronsNumber) == False or \
            self.__isLRegularizationPenaltyValueValid(penaltyValue) == False :
            raise ValueError("Neurons number and regularization penalty cannot be a negative value")
        
        self.__model.add(Dense(neuronsNumber, activation = self.__activationFunctionsButLast,
                                   kernel_regularizer = regularizers.l2(penaltyValue)))
    
    def setEpochsNumber(self, epochsNumber):
        if epochsNumber <= 0 :
            raise ValueError("Epochs number must be a positive value")
                
        self.__epochsNumber = epochsNumber

    def setBatchSize(self, batchSize):
        if batchSize < 0 :
            raise ValueError("Batch size cannot be a negative value")
        
        self.__batchSize = batchSize

    def setDropout(self, dropoutRate: float):
        if dropoutRate < 0 and dropoutRate > 1 :
            raise ValueError("Dropout value must be between 0 and 1")

        self.__model.add(Dropout(dropoutRate))

    def setEarlyStopping(self, patienceValue: int):
        if patienceValue < 0 :
            raise ValueError("Patience value cannot be a negative value")
        
        self.__isEarlyStoppingActivated = True
        self.__earlyStoppingCallback = [EarlyStopping(
            monitor = "val_loss", 
            patience = patienceValue,
            restore_best_weights = True)]

    def removeEarlyStopping(self):
        self.__isEarlyStoppingActivated = False

    def finalizeAdamModel(self, learning_rate):
        self.__model.add(Dense(units = self.__outputSize, activation = "linear"))
        
        self.__model.compile(
            optimizer= Adam(learning_rate = learning_rate),
            loss = self.__nnLoss)
        
        return self        
        
    def _setInputSize(self, inputSize : int):
        self.__inputSize = inputSize

    def _setOutputSize(self, outputSize : int):
        self.__outputSize = outputSize
        
    def __isNeuronsNumberValid(self, neuronsNumber: int):
        return neuronsNumber >= 0
    
    def __isLRegularizationPenaltyValueValid(self, neuronsNumber: float):
        return neuronsNumber >= 0
