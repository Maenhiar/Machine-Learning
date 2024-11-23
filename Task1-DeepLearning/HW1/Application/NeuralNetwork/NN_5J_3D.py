
from Application.NeuralNetwork import NeuralNetwork

class NN_5J_3D(NeuralNetwork):
    __inputNeuronsNumber = 5
    __outputNeuronsNumber = 3
    
    def __init__(self, firstLayerNeuronsNumber : int, inputSet, outputSet):
        super().__init__(firstLayerNeuronsNumber, inputSet, outputSet)