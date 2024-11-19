from CSVParser.CSVParser_3J_2D import CSVParser_3J_2D
from typing import List
import numpy as np

class CSVParser_5J_3D(CSVParser_3J_2D):
    input3ColumnName = " j3"
    input4ColumnName = " j4"
    output2ColumnName = " ee_z"
    
    def __init__(self, filepath: str, separator: str, headerValue: int):
        super().__init__(filepath, separator, headerValue)

    def getInput(self) -> List[List[np.float64]]:
        self.__input = self._parsedCSV[[self.__class__.input0ColumnName, self.__class__.input1ColumnName, 
                                            self.__class__.input2ColumnName, self.__class__.input3ColumnName, 
                                                self.__class__.input4ColumnName]].values
        
        return self.__input

    def getOutput(self) -> List[List[np.float64]]:
        self.__output = self._parsedCSV[[self.__class__.output0ColumnName, self.__class__.output1ColumnName, 
                                            self.__class__.output2ColumnName]].values
        
        return self.__output
