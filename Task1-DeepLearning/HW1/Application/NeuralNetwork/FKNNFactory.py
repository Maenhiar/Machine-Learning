import os
import numpy as np
from CSVParser.CSVParser_2J_2D import CSVParser_2J_2D
from CSVParser.CSVParser_3J_2D import CSVParser_3J_2D
from CSVParser.CSVParser_5J_3D import CSVParser_5J_3D
from NeuralNetwork.NN_2J_2D import NN_2J_2D
from NeuralNetwork.NN_3J_2D import NN_3J_2D
from NeuralNetwork.NN_5J_3D import NN_5J_3D

class FKNNFactory():
    """
    This class implements the Factory Method pattern to provide the correct instantiation
    of the proper neural network model for each of our problems (2J-2D, 3J-2D and 5J-3D).
    
    """
    __parent_dir = os.path.abspath(os.path.join(os.getcwd()))
    __csvTrainingSetFilePath = os.path.join(__parent_dir, "Dataset", "Training-set")
    __csvTestSetFilePath = os.path.join(__parent_dir, "Dataset", "Test-set")

    __trainingSetInputDictionaryKey = "training-set-input"
    __trainingSetOutputDictionaryKey = "training-set-output"
    __testSetInputDictionaryKey = "test-set-input"
    __testSetOutputDictionaryKey = "test-set-output"
    __modelDictionaryKey = "modelWrapper"

    def create2J_2D_NN(self, useStaticSeed: bool):
        csvFilePath = os.path.join(self.__csvTrainingSetFilePath, 'r2', 'r2_21_100k.csv')
        csvParser = CSVParser_2J_2D(csvFilePath)
        partialTrainingSetInput = csvParser.getInput()
        partialTrainingSetOutput = csvParser.getOutput()

        csvFilePath = os.path.join(self.__csvTrainingSetFilePath, 'r2', 'r2_23_100k.csv')
        csvParser = CSVParser_2J_2D(csvFilePath)
        trainingSetInput = np.concatenate((partialTrainingSetInput, csvParser.getInput()), axis=0)
        trainingSetOutput = np.concatenate((partialTrainingSetOutput, csvParser.getOutput()), axis=0)

        csvFilePath = os.path.join(self.__csvTestSetFilePath, 'r2', 'r2_26_100k.csv')
        csvParser = CSVParser_2J_2D(csvFilePath)
        partialTestSetInput = csvParser.getInput()
        partialTestSetOutput = csvParser.getOutput()
        
        csvFilePath = os.path.join(self.__csvTestSetFilePath, 'r2', 'r2_27_100k.csv')
        csvParser = CSVParser_2J_2D(csvFilePath)
        testSetInput = np.concatenate((partialTestSetInput, csvParser.getInput()), axis=0)
        testSetOutput = np.concatenate((partialTestSetOutput, csvParser.getOutput()), axis=0)

        modelData = {
            self.__trainingSetInputDictionaryKey: trainingSetInput,
            self.__trainingSetOutputDictionaryKey: trainingSetOutput,
            self.__testSetInputDictionaryKey: testSetInput,
            self.__testSetOutputDictionaryKey: testSetOutput,
            self.__modelDictionaryKey: NN_2J_2D(useStaticSeed)
        }

        return modelData

    def create3J_2D_NN(self, useStaticSeed: bool):
        csvFilePath = os.path.join(self.__csvTrainingSetFilePath, 'r3', 'r3_24_100k.csv')
        csvParser = CSVParser_3J_2D(csvFilePath)
        partialTrainingSetInput = csvParser.getInput()
        partialTrainingSetOutput = csvParser.getOutput()

        csvFilePath = os.path.join(self.__csvTrainingSetFilePath, 'r3', 'r3_28_100k.csv')
        csvParser = CSVParser_3J_2D(csvFilePath)
        trainingSetInput = np.concatenate((partialTrainingSetInput, csvParser.getInput()), axis=0)
        trainingSetOutput = np.concatenate((partialTrainingSetOutput, csvParser.getOutput()), axis=0)

        csvFilePath = os.path.join(self.__csvTestSetFilePath, 'r3', 'r3_20_100k.csv')
        csvParser = CSVParser_3J_2D(csvFilePath)
        partialTestSetInput = csvParser.getInput()
        partialTestSetOutput = csvParser.getOutput()

        csvFilePath = os.path.join(self.__csvTestSetFilePath, 'r3', 'r3_29_100k.csv')
        csvParser = CSVParser_3J_2D(csvFilePath)
        testSetInput = np.concatenate((partialTestSetInput, csvParser.getInput()), axis=0)
        testSetOutput = np.concatenate((partialTestSetOutput, csvParser.getOutput()), axis=0)
        
        modelData = {
            self.__trainingSetInputDictionaryKey: trainingSetInput,
            self.__trainingSetOutputDictionaryKey: trainingSetOutput,
            self.__testSetInputDictionaryKey: testSetInput,
            self.__testSetOutputDictionaryKey: testSetOutput,
            self.__modelDictionaryKey: NN_3J_2D(useStaticSeed)
        }

        return modelData

    def create5J_3D_NN(self, useStaticSeed: bool):
        csvFilePath = os.path.join(self.__csvTrainingSetFilePath, 'r5', 'r5_25_100k.csv')
        csvParser = CSVParser_5J_3D(csvFilePath)
        partialTrainingSetInput = csvParser.getInput()
        partialTrainingSetOutput = csvParser.getOutput()

        csvFilePath = os.path.join(self.__csvTrainingSetFilePath, 'r5', 'r5_30_100k.csv')
        csvParser = CSVParser_5J_3D(csvFilePath)
        trainingSetInput = np.concatenate((partialTrainingSetInput, csvParser.getInput()), axis=0)
        trainingSetOutput = np.concatenate((partialTrainingSetOutput, csvParser.getOutput()), axis=0)

        csvFilePath = os.path.join(self.__csvTestSetFilePath, 'r5', 'r5_22_100k.csv')
        csvParser = CSVParser_5J_3D(csvFilePath)
        partialTestSetInput = csvParser.getInput()
        partialTestSetOutput = csvParser.getOutput()

        csvFilePath = os.path.join(self.__csvTestSetFilePath, 'r5', 'r5_27_100k.csv')
        csvParser = CSVParser_5J_3D(csvFilePath)
        testSetInput = np.concatenate((partialTestSetInput, csvParser.getInput()), axis=0)
        testSetOutput = np.concatenate((partialTestSetOutput, csvParser.getOutput()), axis=0)
        
        modelData = {
            self.__trainingSetInputDictionaryKey: trainingSetInput,
            self.__trainingSetOutputDictionaryKey: trainingSetOutput,
            self.__testSetInputDictionaryKey: testSetInput,
            self.__testSetOutputDictionaryKey: testSetOutput,
            self.__modelDictionaryKey: NN_5J_3D(useStaticSeed)
        }

        return modelData