from NeuralNetwork.NeuralNetwork import NeuralNetwork


class AverageModelProperties():
    """
    This class represents a model along with its properties and performance parameters.
    This class is needed due to impossibility of correctly creating a clone 
    of a keras model without incurring in any kind of error or problem.
    """
    __model = None
    __finalMSE = None
    __finalTestSetMSE = None
    __avgTrainingTime = 0
    __modelHistories = None
    __fkNNModel = None

    def __init__(self, model, finalMSE, finalTestSetMSE, avgTrainingTime, modelHistories):
        if not isinstance(model, NeuralNetwork):
            raise TypeError(f"The model is not an instance of a Neural Network class")

        self.__fkNNModel = model
        self.__model = model._getModel()
        self.__finalMSE = finalMSE
        self.__finalTestSetMSE = finalTestSetMSE
        self.__avgTrainingTime = avgTrainingTime
        self.__modelHistories = modelHistories

    def getFinalMSE(self):
        return self.__finalMSE

    def getFinalTestSetMSE(self):
        return self.__finalTestSetMSE
    
    def getTrainingTime(self):
        return self.__avgTrainingTime

    def getModelHistory(self):
        return self.__model.history

    def getEarlyStopping(self):
        return self.__fkNNModel.getEarlyStopping()
    
    def getModelLayers(self):
        return self.__model.layers
    
    def getModelLoss(self):
        return self.__model.loss
    
    def getOptimizerConfiguration(self):
        return self.__model.optimizer.get_config()
    
    def getFKNNModel(self):
        return self.__fkNNModel
    
    def getmodelHistories(self):
        return self.__modelHistories
    
    def showModelSummary(self):
        """ Necessary due to impossibility of correctly creating a clone 
        of a keras model without incurring in any kind of error or problem."""
        self.__model.summary()

