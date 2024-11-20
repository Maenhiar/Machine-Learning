import numpy as np
from typing import List
import pandas as pd

class CSVParser_2J_2D():
    __input0ColumnName = "j0"
    __input1ColumnName = " j1"
    __output0ColumnName = " ee_x"
    __output1ColumnName = " ee_y"
        
    def __init__(self, filepath: str, separator: str, headerValue: int):     
        self.__filepath = filepath
        self.__separator = separator
        self.__headerValue = headerValue
        
        try:
            self._parsedCSV = pd.read_csv(self.__filepath, sep = self.__separator, header = self.__headerValue)
        except Exception as e:
            print(f"{e}")
            raise
        
    def getInput(self) -> List[List[np.float64]]:
        self.__input = self._parsedCSV[[self.__input0ColumnName, self.__input1ColumnName]].values
        
        return self.__input

    def getOutput(self) -> List[List[np.float64]]:
        self.__output = self._parsedCSV[[self.__output0ColumnName, self.__output1ColumnName]].values
        
        return self.__output
    
    def __str__(self) -> str:
        
        return f"""CSVParser [filepath={self.__filepath}, separator={self.__separator}, headerValue={self.__headerValue}, 
            input={self.__input},   output={self.__output}]"""

    def _getInput0ColumnName(self) -> str:
        return self.__input0ColumnName
    
    def _getInput1ColumnName(self) -> str:
        return self.__input1ColumnName
    
    def _getOutput0ColumnName(self) -> str:
        return self.__output0ColumnName
    
    def _getOutput1ColumnName(self) -> str:
        return self.__output1ColumnName