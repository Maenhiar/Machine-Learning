from CSVParser.CSVParser_2J_2D import CSVParser_2J_2D
from CSVParser.CSVParser_3J_2D import CSVParser_3J_2D
from CSVParser.CSVParser_5J_3D import CSVParser_5J_3D
from NeuralNetwork.NN_2J_2D import NN_2J_2D
from NeuralNetwork.NN_3J_2D import NN_3J_2D
from NeuralNetwork.NN_5J_3D import NN_5J_3D

class FKNNFactory():

    def create2J_2D_NN(self):
        csvParser = CSVParser_2J_2D()
        inputSet = csvParser.getInput()
        outputSet = csvParser.getOutput()
        return NN_2J_2D(inputSet, outputSet)

    def create3J_2D_NN(self):
        csvParser = CSVParser_3J_2D()
        inputSet = csvParser.getInput()
        outputSet = csvParser.getOutput()
        return NN_3J_2D(inputSet, outputSet)

    def create5J_3D_NN(self):
        csvParser = CSVParser_5J_3D()
        inputSet = csvParser.getInput()
        outputSet = csvParser.getOutput()
        return NN_5J_3D(inputSet, outputSet)