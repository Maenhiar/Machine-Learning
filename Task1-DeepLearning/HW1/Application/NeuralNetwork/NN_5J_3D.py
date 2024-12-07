from PerformaceEvaluation.KFoldCrossValidation import KFoldCrossValidation
from NeuralNetwork.NeuralNetwork import NeuralNetwork

class NN_5J_3D(NeuralNetwork):
    """
    This class provides the common features to instantiate a Keras
    feedforward neural network model for a 5J-3D forward kinematics problem, ensuring 
    no errors in initialization or components addition. It provides methods that 
    allow for easily adding and retrieving the neural network's components, such as 
    layers, neurons, dropout, epochs, batch size, dropout, L1 or L2 regularization and early stopping settings."
    This class also ensures that hidden layers activation function is always ReLU and 
    output layer activation function is always linear because this is the 
    most common efficient choice in regression problems tackled with a feedforward neural networ.
    The chosen loss function is Mean Squared Error and the optimizer is Adam with a configurable learning rate.
    """
    def __init__(self, useStaticSeed: bool):
        self._setInputSize(5)
        self._setOutputSize(3)
        super().__init__(useStaticSeed)