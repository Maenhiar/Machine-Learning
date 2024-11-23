from NeuralNetwork.NeuralNetwork import NeuralNetwork

class NN_2J_2D(NeuralNetwork):

    def __init__(self, inputSet, outputSet):
        self._setInputNeuronsNumber(2)
        self._setOutputNeuronsNumber(2)
        super().__init__(inputSet, outputSet)