from abc import ABC
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
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
    
    def __init__(self):
        if type(self) is NeuralNetwork:
            raise NotImplementedError("Cannot instance an abstract class.")
        
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
