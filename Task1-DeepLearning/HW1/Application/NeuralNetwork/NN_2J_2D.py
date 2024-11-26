from PerformaceEvaluation.KFoldCrossValidation import KFoldCrossValidation
from NeuralNetwork.NeuralNetwork import NeuralNetwork

class NN_2J_2D(NeuralNetwork):

    def __init__(self, trainingSetInput, trainingSetOutput, testSetInput, testSetOutput):
        self._setInputNeuronsNumber(2)
        self._setOutputNeuronsNumber(2)
        super().__init__(trainingSetInput, trainingSetOutput, testSetInput, testSetOutput)