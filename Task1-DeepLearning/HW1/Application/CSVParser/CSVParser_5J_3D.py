import os
from CSVParser.CSVParser_3J_2D import CSVParser_3J_2D
from typing import List
import numpy as np

class CSVParser_5J_3D(CSVParser_3J_2D):
    """
    This class parses the 5J-3D CSV file passed as input and provides the associated
    input set and output set.

    Methods:
        getInput(): Returns the training set inputs.
        getOutput(): Returns the training set outputs.
    """
    __input3ColumnName = " j3"
    __input4ColumnName = " j4"
    __output2ColumnName = " ee_z"
    
    def __init__(self, csvFilePath):
        super().__init__(csvFilePath)

    def getInput(self) -> List[List[np.float64]]:
        input = self._getParsedCSV()[[self._getInput0ColumnName(), self._getInput1ColumnName(), 
                                            self._getInput2ColumnName(), self.__input3ColumnName, 
                                                self.__input4ColumnName]].values
        
        return input

    def getOutput(self) -> List[List[np.float64]]:
        output = self._getParsedCSV()[[self._getOutput0ColumnName(), self._getOutput1ColumnName(), 
                                            self.__output2ColumnName]].values
        
        return output
