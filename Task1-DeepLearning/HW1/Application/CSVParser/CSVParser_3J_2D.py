from CSVParser.CSVParser_2J_2D import CSVParser_2J_2D
from typing import List
import numpy as np

class CSVParser_3J_2D(CSVParser_2J_2D):
    input2ColumnName = " j2"
    
    def __init__(self, filepath: str, separator: str, headerValue: int):
        super().__init__(filepath, separator, headerValue)

    def getInput(self) -> List[List[np.float64]]:
        self.__input = self._parsedCSV[[self.__class__.input0ColumnName, self.__class__.input1ColumnName, 
                                         self.__class__.input2ColumnName]].values
        
        return self.__input

    def getOutput(self) -> List[List[np.float64]]:
        self.__output = self._parsedCSV[[self.__class__.output0ColumnName, self.__class__.output1ColumnName]].values
        
        return self.__output
