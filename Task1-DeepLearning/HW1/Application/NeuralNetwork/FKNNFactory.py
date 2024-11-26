import os
from CSVParser.CSVParser_2J_2D import CSVParser_2J_2D
from CSVParser.CSVParser_3J_2D import CSVParser_3J_2D
from CSVParser.CSVParser_5J_3D import CSVParser_5J_3D
from NeuralNetwork.NN_2J_2D import NN_2J_2D
from NeuralNetwork.NN_3J_2D import NN_3J_2D
from NeuralNetwork.NN_5J_3D import NN_5J_3D
from PerformaceEvaluation.KFoldCrossValidation import KFoldCrossValidation

class FKNNFactory():
    __parent_dir = os.path.abspath(os.path.join(os.getcwd()))
    __csvTrainingSetFilePath = os.path.join(__parent_dir, "Dataset", "Training-set")
    __csvTestSetFilePath = os.path.join(__parent_dir, "Dataset", "Test-set")

    def create2J_2D_NN(self):
        __csvTrainingSetFilePath = os.path.join(self.__csvTrainingSetFilePath, 'r2', 'r2_21_100k.csv')
        csvParser = CSVParser_2J_2D(__csvTrainingSetFilePath)
        trainingSetInput = csvParser.getInput()
        trainingSetOutput = csvParser.getOutput()

        __csvTestSetFilePath = os.path.join(self.__csvTestSetFilePath, 'r2', 'r2_26_100k.csv')
        csvParser = CSVParser_2J_2D(__csvTestSetFilePath)
        testSetInput = csvParser.getInput()
        testSetOutput = csvParser.getOutput()
        return NN_2J_2D(trainingSetInput, trainingSetOutput, testSetInput, testSetOutput)

    def create3J_2D_NN(self):
        __csvTrainingSetFilePath = os.path.join(self.__csvTrainingSetFilePath, 'r3', 'r3_24_100k.csv')
        csvParser = CSVParser_3J_2D(__csvTrainingSetFilePath)
        trainingSetInput = csvParser.getInput()
        trainingSetOutput = csvParser.getOutput()

        __csvTestSetFilePath = os.path.join(self.__csvTestSetFilePath, 'r3', 'r3_20_100k.csv')
        csvParser = CSVParser_3J_2D(__csvTestSetFilePath)
        testSetInput = csvParser.getInput()
        testSetOutput = csvParser.getOutput()
        return NN_3J_2D(trainingSetInput, trainingSetOutput, testSetInput, testSetOutput)

    def create5J_3D_NN(self):
        __csvTrainingSetFilePath = os.path.join(self.__csvTrainingSetFilePath, 'r5', 'r5_25_100k.csv')
        csvParser = CSVParser_5J_3D(__csvTrainingSetFilePath)
        trainingSetInput = csvParser.getInput()
        trainingSetOutput = csvParser.getOutput()

        __csvTestSetFilePath = os.path.join(self.__csvTestSetFilePath, 'r5', 'r5_22_100k.csv')
        csvParser = CSVParser_5J_3D(__csvTestSetFilePath)
        testSetInput = csvParser.getInput()
        testSetOutput = csvParser.getOutput()
        return NN_5J_3D(trainingSetInput, trainingSetOutput, testSetInput, testSetOutput)