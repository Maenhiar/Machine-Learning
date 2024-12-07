import os
from CSVParser.CSVParser_2J_2D import CSVParser_2J_2D
from typing import List
import numpy as np

class CSVParser_3J_2D(CSVParser_2J_2D):
    """
    This class parses the 3J-2D CSV file passed as input and provides the associated
    input set and output set.

    Methods:
        getInput(): Returns the training set inputs.
        getOutput(): Returns the training set outputs.
    """
    __input2ColumnName = " j2"
    
    def __init__(self, csvFilePath):
        super().__init__(csvFilePath)

    def getInput(self) -> List[List[np.float64]]:
        input = self._getParsedCSV()[[self._getInput0ColumnName(), self._getInput1ColumnName(), 
                                         self.__input2ColumnName]].values
        
        return input

    def getOutput(self) -> List[List[np.float64]]:
        output = self._getParsedCSV()[[self._getOutput0ColumnName(), self._getOutput1ColumnName()]].values
        
        return output
    
    def _getInput2ColumnName(self) -> str:
        return self.__input2ColumnName
