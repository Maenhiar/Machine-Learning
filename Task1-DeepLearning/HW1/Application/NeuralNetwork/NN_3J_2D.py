from NeuralNetwork.NeuralNetwork import NeuralNetwork

class NN_3J_2D(NeuralNetwork):
    
    def __init__(self, inputSet, outputSet):
        self._setInputNeuronsNumber(3)
        self._setOutputNeuronsNumber(2)
        super().__init__(inputSet, outputSet)