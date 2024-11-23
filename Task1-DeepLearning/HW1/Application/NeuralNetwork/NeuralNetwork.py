from abc import ABC
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers

class NeuralNetwork(ABC):
    __inputNeuronsNumber = 2
    __outputNeuronsNumber = 2
    __batch_size = 32
    __epochsNumber = 10
    __inputSet = None
    __outputSet = None
    __trainsetInput = None
    __trainsetOutput = None
    __testsetInput = None
    __testsetOutput = None 
    __validationsetInput = None
    __validationsetOutput = None
    __inputLayer = None
    __modelHiddenLayers = None
    __model = None
    __isEarlyStoppingActivated = False
    __earlyStoppingCallback = None
    __isFirstHiddenLayer = True
    
    def __init__(self, inputSet, outputSet):
        if type(self) is NeuralNetwork:
            raise NotImplementedError("Non è possibile creare un'istanza della classe astratta!")
        
        self.__inputSet = inputSet
        self.__outputSet = outputSet
        self.__inputLayer = Input(shape=(self.__inputNeuronsNumber,))
        self.__trainTestValidationSetsSetup()

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
        
        self.__batch_size = batchSize

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

        print(self.__model)

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
        print(self.__model.summary())

    def __trainTestValidationSetsSetup(self) :
        trainsetProportion = int(2 * len(self.__inputSet) / 3)  # 2/3 per il train
        self.__trainsetInput, self.__trainsetOutput = self.__inputSet[:trainsetProportion], self.__outputSet[:trainsetProportion]
        self.__testsetInput, self.__testsetOutput = self.__inputSet[trainsetProportion:], self.__outputSet[trainsetProportion:] 

        self.__trainsetInput = self.__trainsetInput.astype("float32")
        self.__testsetInput = self.__testsetInput.astype("float32")

        self.__trainsetOutput = self.__trainsetOutput.astype("float32")
        self.__testsetOutput = self.__testsetOutput.astype("float32")
        
        sampleToReserve = 10000
        self.__validationsetInput = self.__trainsetInput[-sampleToReserve:]
        self.__validationsetOutput = self.__trainsetOutput[-sampleToReserve:]
        self.__trainsetInput = self.__trainsetInput[:-sampleToReserve]
        self.__trainsetOutput = self.__trainsetOutput[:-sampleToReserve]

    def _setInputNeuronsNumber(self, inputNeuronsNumber : int):
        self.__inputNeuronsNumber = inputNeuronsNumber

    def _setOutputNeuronsNumber(self, outputNeuronsNumber : int):
        self.__outputNeuronsNumber = outputNeuronsNumber
        
    def __fit(self):
        print("Fit model on training data")
        history = self.__model.fit(
            self.__trainsetInput,
            self.__trainsetOutput,
            batch_size = self.__batch_size,
            epochs = self.__epochsNumber,
            callbacks = self.__earlyStoppingCallback if self.__isEarlyStoppingActivated else [],
            validation_data = (self.__validationsetInput, self.__validationsetOutput),
        )

        print(history.history)

        print("Evaluate on test data")
        results = self.__model.evaluate(self.__testsetInput, self.__testsetOutput, batch_size = self.__batch_size)
        print("test loss, test acc:", results)

        print("Generate predictions for 3 samples")
        predictions = self.__model(self.__testsetInput).numpy()
        print("predictions shape:", predictions.shape)

        r2 = r2_score(self.__testsetOutput, predictions)
        print(f"R-squared: {r2}")

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
