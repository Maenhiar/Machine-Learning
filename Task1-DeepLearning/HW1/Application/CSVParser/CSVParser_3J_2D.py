from CSVParser.CSVParser_2J_2D import CSVParser_2J_2D
from typing import List
import numpy as np

class CSVParser_3J_2D(CSVParser_2J_2D):
    __input2ColumnName = " j2"
    
    def __init__(self, filepath: str, separator: str, headerValue: int):
        super().__init__(filepath, separator, headerValue)

    def getInput(self) -> List[List[np.float64]]:
        self.__input = self._parsedCSV[[self._getInput0ColumnName(), self._getInput1ColumnName(), 
                                         self.__input2ColumnName]].values
        
        return self.__input

    def getOutput(self) -> List[List[np.float64]]:
        self.__output = self._parsedCSV[[self._getOutput0ColumnName(), self._getOutput1ColumnName()]].values
        
        return self.__output
    
    def _getInput2ColumnName(self) -> str:
        return self.__input2ColumnName
