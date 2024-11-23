import os
from CSVParser.CSVParser_2J_2D import CSVParser_2J_2D
from typing import List
import numpy as np

class CSVParser_3J_2D(CSVParser_2J_2D):
    __input2ColumnName = " j2"
    
    def __init__(self):
        super().__init__()

    def getInput(self) -> List[List[np.float64]]:
        input = self._getParsedCSV()[[self._getInput0ColumnName(), self._getInput1ColumnName(), 
                                         self.__input2ColumnName]].values
        
        return input

    def getOutput(self) -> List[List[np.float64]]:
        output = self._getParsedCSV()[[self._getOutput0ColumnName(), self._getOutput1ColumnName()]].values
        
        return output
    
    def _getInput2ColumnName(self) -> str:
        return self.__input2ColumnName
    
    def _setCSVFileFinalPath(self) :
        self._setCSVFilePath(os.path.join(self._getCSVFilePath(), 'r3', 'r3_24_100k.csv'))
        return;
