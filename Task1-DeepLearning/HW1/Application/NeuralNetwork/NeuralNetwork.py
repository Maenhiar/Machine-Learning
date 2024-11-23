from abc import ABC
from sklearn.metrics import r2_score
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD

class NeuralNetwork(ABC):
    __inputNeuronsNumber = 2
    __outputNeuronsNumber = 2
    __batch_size = 32
    __epochsNumber = 10
    __neuronsNumber = 1
    __inputSet = None
    __outputSet = None
    __trainsetInput = None
    __trainsetOutput = None
    __testsetInput = None
    __testsetOutput = None 
    __validationsetInput = None
    __validationsetOutput = None
    
    def __init__(self, firstLayerNeuronsNumber : int, inputSet, outputSet):
        if type(self) is NeuralNetwork:
            raise NotImplementedError("Non è possibile creare un'istanza della classe astratta!")

        if self.__isNeuronsNumberValid(firstLayerNeuronsNumber) == False :
            raise ValueError("Il valore deve essere maggiore o uguale a 0")

        self.__inputSet = inputSet
        self.__outputSet = outputSet
        self.__neuronsNumber = firstLayerNeuronsNumber
        self.__modelSetup()
        self.__trainTestValidationSetsSetup()

    def addDenseLayer(self, neuronsNumber: int):
        if self.__isNeuronsNumberValid(neuronsNumber) == False :
            return
        
        self.__modelLayers = Dense(64, activation="relu")(self.__modelLayers)

    def addDenseLayerWithL1Regularization(self, neuronsNumber: int, penaltyValue = float):
        if self.__isNeuronsNumberValid(neuronsNumber) == False or \
            self.__isLRegularizationPenaltyValueValid(penaltyValue) == False :
            return
        
        self.__modelLayers = Dense(neuronsNumber, activation="relu", 
                                   kernel_regularizer=regularizers.l1(penaltyValue))(self.__modelLayers)

    def addDenseLayerWithL2Regularization(self, neuronsNumber: int, penaltyValue = float):
        if self.__isNeuronsNumberValid(neuronsNumber) == False or \
            self.__isLRegularizationPenaltyValueValid(penaltyValue) == False :
            return
        self.__modelLayers = Dense(neuronsNumber, activation="relu", 
                                   kernel_regularizer=regularizers.l2(penaltyValue))(self.__modelLayers)
    
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
        
        self.__modelLayers = Dropout(dropoutRate)(self.__modelLayers)

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

    def __modelSetup(self):
        inputLayer = Input(shape=(self.__inputNeuronsNumber,))
        self.__modelLayers = Dense(self.__neuronsNumber, activation="relu")(inputLayer)
        outputLayer = Dense(self.__outputNeuronsNumber, activation="linear")(self.__modelLayers)
        self.__model = Model(inputs = inputLayer, outputs = outputLayer)
        print(self.__model.summary())

    def __trainTestValidationSetsSetup(self) :
        trainsetProportion = int(2 * len(self.__inputSet) / 3)  # 2/3 per il train
        self.__trainsetInput, self.__trainsetOutput = input[:trainsetProportion], self.__outputSet[:trainsetProportion]
        self.__testsetInput, self.__testsetOutput = input[trainsetProportion:], self.__outputSet[trainsetProportion:] 

        self.__trainsetInput = self.__trainsetInput.astype("float32")
        testsetInput = testsetInput.astype("float32")

        self.__trainsetOutput = self.__trainsetOutput.astype("float32")
        testsetOutput = testsetOutput.astype("float32")

        print(testsetInput.shape)

        sampleToReserve = 10000
        self.__validationsetInput = self.__trainsetInput[-sampleToReserve:]
        self.__validationsetOutput = self.__trainsetOutput[-sampleToReserve:]
        self.__trainsetInput = self.__trainsetInput[:-sampleToReserve]
        self.__trainsetOutput = self.__trainsetOutput[:-sampleToReserve]
    
    def fitAdam(self, learning_rate):
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
        
        self.__model.compile(
            optimizer = SGD(learning_rate = learning_rate, momentum = sgdMomentumValue), 
            loss = "mean_squared_error", 
            metrics = ["mean_squared_error"])
        
        self.__fit()
        
    def __fit(self):
        print("Fit model on training data")
        history = self.__model.fit(
            self.__trainsetInput,
            self.__trainsetOutput,
            batch_size = self.__batch_size,
            epochs = self.__epochsNumber,
            callbacks = [self.__earlyStoppingCallback],
            validation_data=(self.__validationsetInput, self.__validationsetOutput),
        )

        print(history.history)

        print("Evaluate on test data")
        results = self.__model.evaluate(self.__testsetInput, self.__testsetOutput, batch_size = self.__batch_size)
        print("test loss, test acc:", results)

        print("Generate predictions for 3 samples")
        predictions = self.__model(self.__testsetInput)
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
