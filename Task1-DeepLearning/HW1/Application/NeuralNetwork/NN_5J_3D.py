from PerformaceEvaluation.KFoldCrossValidation import KFoldCrossValidation
from NeuralNetwork.NeuralNetwork import NeuralNetwork

class NN_5J_3D(NeuralNetwork):

    def __init__(self, kfoldCrossValidation : KFoldCrossValidation):
        self._setInputNeuronsNumber(5)
        self._setOutputNeuronsNumber(3)
        super().__init__(kfoldCrossValidation)