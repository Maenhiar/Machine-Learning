
from Application.NeuralNetwork import NeuralNetwork

class NN_2J_2D(NeuralNetwork):
    __inputNeuronsNumber = 2
    __outputNeuronsNumber = 2
    
    def __init__(self, firstLayerNeuronsNumber : int, inputSet, outputSet):
        super().__init__(firstLayerNeuronsNumber, inputSet, outputSet)