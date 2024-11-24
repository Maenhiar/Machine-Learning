from abc import ABC
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
from PerformaceEvaluation.KFoldCrossValidation import KFoldCrossValidation

class NeuralNetwork(ABC):
    __inputNeuronsNumber = 2
    __outputNeuronsNumber = 2
    __batchSize = 32
    __epochsNumber = 10
    __inputLayer = None
    __modelHiddenLayers = None
    __model = None
    __isEarlyStoppingActivated = False
    __earlyStoppingCallback = None
    __isFirstHiddenLayer = True
    __kfoldCrossValidation = None
    
    def __init__(self, kfoldCrossValidation : KFoldCrossValidation):
        if type(self) is NeuralNetwork:
            raise NotImplementedError("Non è possibile creare un'istanza della classe astratta!")
        
        if kfoldCrossValidation is None:
            raise ValueError("L'oggetto non può essere None.")
        
        self.__kfoldCrossValidation = kfoldCrossValidation
        self.__inputLayer = Input(shape=(self.__inputNeuronsNumber,))

    def addDenseLayer(self, neuronsNumber: int):
        if self.__isNeuronsNumberValid(neuronsNumber) == False :
            return
        
        if self.__isFirstHiddenLayer :
            self.__isFirstHiddenLayer = False
            self.__modelHiddenLayers = Dense(neuronsNumber, activation="relu")(self.__inputLayer)
        else :
            self.__modelHiddenLayers = Dense(neuronsNumber, activation="relu")(self.__modelHiddenLayers)

    def addDenseLayerWithL1Regularization(self, neuronsNumber: int, penaltyValue = float):
        if self.__isNeuronsNumberValid(neuronsNumber) == False or \
            self.__isLRegularizationPenaltyValueValid(penaltyValue) == False :
            return
        
        self.__modelHiddenLayers = Dense(neuronsNumber, activation="relu", 
                                   kernel_regularizer=regularizers.l1(penaltyValue))(self.__modelHiddenLayers)

    def addDenseLayerWithL2Regularization(self, neuronsNumber: int, penaltyValue = float):
        if self.__isNeuronsNumberValid(neuronsNumber) == False or \
            self.__isLRegularizationPenaltyValueValid(penaltyValue) == False :
            return
        self.__modelHiddenLayers = Dense(neuronsNumber, activation="relu", 
                                   kernel_regularizer=regularizers.l2(penaltyValue))(self.__modelHiddenLayers)
    
    def setEpochsNumber(self, epochsNumber):
        if self.__isEpochsNumberValid(epochsNumber) == False :
            return
        
        self.__epochsNumber = epochsNumber

    def setBatchSize(self, batchSize):
        if self.__isBatchSizeValid(batchSize) == False :
            return
        
        self.__batchSize = batchSize

    def setDropout(self, dropoutRate: float):
        if self.__isDropoutRateValueValid(dropoutRate) == False :
            return
        
        self.__modelHiddenLayers = Dropout(dropoutRate)(self.__modelHiddenLayers)

    def setEarlyStopping(self, patienceValue: int):
        if self.__isEarlystoppingPatienceValueValid(patienceValue) == False :
            return
        
        self.__isEarlyStoppingActivated = True
        self.__earlyStoppingCallback = [EarlyStopping(
            monitor='val_loss', 
            patience = patienceValue,
            restore_best_weights = True
        )] if self.__isEarlyStoppingActivated else []

    def removeEarlyStopping(self):
        self.__isEarlyStoppingActivated = False

    def fitAdam(self, learning_rate):
        self.__modelSetup()
        
        self.__model.compile(
            optimizer= Adam(learning_rate = learning_rate),
            loss = "mean_squared_error",
            metrics = ["mean_squared_error"],
        )

        self.__fit()

    def fitSGD(self, learning_rate, sgdMomentumValue):
        if self.__isSGDLearningRateValid(learning_rate) == False or \
            self.__isSGDMomentumValid(sgdMomentumValue) == False :
            return
        
        self.__modelSetup()
        
        self.__model.compile(
            optimizer = SGD(learning_rate = learning_rate, momentum = sgdMomentumValue), 
            loss = "mean_squared_error", 
            metrics = ["mean_squared_error"])
        
        self.__fit()

    def __modelSetup(self):
        outputLayer = Dense(self.__outputNeuronsNumber, activation="linear")(self.__modelHiddenLayers)
        self.__model = Model(inputs = self.__inputLayer, outputs = outputLayer)
        
    def _setInputNeuronsNumber(self, inputNeuronsNumber : int):
        self.__inputNeuronsNumber = inputNeuronsNumber

    def _setOutputNeuronsNumber(self, outputNeuronsNumber : int):
        self.__outputNeuronsNumber = outputNeuronsNumber
        
    def __fit(self):
        self.__kfoldCrossValidation.performKFoldCrossValidation(self.__model, self.__batchSize, self.__epochsNumber, 
                                                                    self.__earlyStoppingCallback)

    def __isNeuronsNumberValid(self, neuronsNumber: int):
        return neuronsNumber >= 0
    
    def __isLRegularizationPenaltyValueValid(self, neuronsNumber: float):
        return neuronsNumber >= 0
    
    def __isDropoutRateValueValid(self, dropoutRate: float):
        return dropoutRate >= 0 and dropoutRate <= 1
    
    def __isEarlystoppingPatienceValueValid(self, patienceValue: int):
        return patienceValue >= 0
    
    def __isBatchSizeValid(self, batchSize : int):
        return batchSize >= 0
    
    def __isEpochsNumberValid(self, epochsNumber : int):
        return epochsNumber > 0
    
    def __isSGDLearningRateValid(self, sgdLearningRate : float):
        return sgdLearningRate > 0
    
    def __isSGDMomentumValid(self, sgdMomentumValue : float):
        return sgdMomentumValue >= 0 and sgdMomentumValue <= 1
