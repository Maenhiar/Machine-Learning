from PerformaceEvaluation.KFoldCrossValidation import KFoldCrossValidation
from NeuralNetwork.NeuralNetwork import NeuralNetwork

class NN_5J_3D(NeuralNetwork):

    def __init__(self, trainingSetInput, trainingSetOutput, testSetInput, testSetOutput):
        self._setInputNeuronsNumber(5)
        self._setOutputNeuronsNumber(3)
        super().__init__(trainingSetInput, trainingSetOutput, testSetInput, testSetOutput)