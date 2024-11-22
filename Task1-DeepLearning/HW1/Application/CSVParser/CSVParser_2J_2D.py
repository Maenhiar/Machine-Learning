import os
import numpy as np
from typing import List
import pandas as pd

from Application.CSVParser import CSVParser

class CSVParser_2J_2D(CSVParser):
    __script_dir = os.path.dirname(os.path.abspath(__file__))
    __parent_dir = os.path.dirname(__script_dir)
    __csv_file_path = os.path.join(__parent_dir, "Dataset", "Training-set")
    __input0ColumnName = "j0"
    __input1ColumnName = " j1"
    __output0ColumnName = " ee_x"
    __output1ColumnName = " ee_y"
    __separator = ";"
    __headerValue = 0
        
    def __init__(self):
        self.__setCSVFileFinalPath(self.__csv_file_path)
        
        try:
            self._parsedCSV = pd.read_csv(self.__csv_file_path, sep = self.__separator, header = self.__headerValue)
        except Exception as e:
            print(f"{e}")
            raise
        
    def getInput(self) -> List[List[np.float64]]:
        self.__input = self._parsedCSV[[self.__input0ColumnName, self.__input1ColumnName]].values
        return self.__input

    def getOutput(self) -> List[List[np.float64]]:
        self.__output = self._parsedCSV[[self.__output0ColumnName, self.__output1ColumnName]].values
        return self.__output
    
    def _getInput0ColumnName(self) -> str:
        return self.__input0ColumnName
    
    def _getInput1ColumnName(self) -> str:
        return self.__input1ColumnName
    
    def _getOutput0ColumnName(self) -> str:
        return self.__output0ColumnName
    
    def _getOutput1ColumnName(self) -> str:
        return self.__output1ColumnName
        
    def __setCSVFileFinalPath(self, __csv_file_path : str) :
        self.__csv_file_path = os.path.join(self.__csv_file_path, 'r2', 'r2_21_100k.csv')
        return;
