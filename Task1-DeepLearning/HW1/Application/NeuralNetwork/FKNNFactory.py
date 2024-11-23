from Application.CSVParser import CSVParser_2J_2D, CSVParser_3J_2D, CSVParser_5J_3D
from Application.NeuralNetwork import NN_2J_2D, NN_3J_2D, NN_5J_3D

class FKNNFactory():

    def create2J_2D_NN(self, firstLayerNeuronsNumber : int):
        csvParser = CSVParser_2J_2D()
        inputSet = csvParser.getInput()
        outputSet = csvParser.getOutput()
        return NN_2J_2D(firstLayerNeuronsNumber, inputSet, outputSet)

    def create3J_2D_NN(self, firstLayerNeuronsNumber : int):
        csvParser = CSVParser_3J_2D()
        inputSet = csvParser.getInput()
        outputSet = csvParser.getOutput()
        return NN_2J_2D(firstLayerNeuronsNumber, inputSet, outputSet)

    def create5J_3D_NN(self, firstLayerNeuronsNumber : int):
        csvParser = CSVParser_5J_3D()
        inputSet = csvParser.getInput()
        outputSet = csvParser.getOutput()
        return NN_5J_3D(firstLayerNeuronsNumber, inputSet, outputSet)