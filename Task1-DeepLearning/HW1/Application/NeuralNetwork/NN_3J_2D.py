
from Application.NeuralNetwork import NeuralNetwork

class NN_3J_2D(NeuralNetwork):
    __inputNeuronsNumber = 3
    __outputNeuronsNumber = 2
    
    def __init__(self, firstLayerNeuronsNumber : int, inputSet, outputSet):
        super().__init__(firstLayerNeuronsNumber, inputSet, outputSet)