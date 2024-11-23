import os
import numpy as np
from typing import List
import pandas as pd

class CSVParser_2J_2D():
    __parent_dir = os.path.abspath(os.path.join(os.getcwd()))
    __csv_file_path = os.path.join(__parent_dir, "Dataset", "Training-set")
    __input0ColumnName = "j0"
    __input1ColumnName = " j1"
    __output0ColumnName = " ee_x"
    __output1ColumnName = " ee_y"
    __separator = ";"
    __headerValue = 0
    __parsedCSV = None
        
    def __init__(self):
        self._setCSVFileFinalPath()
        
        try:
            self.__parsedCSV = pd.read_csv(self.__csv_file_path, sep = self.__separator, header = self.__headerValue)
        except Exception as e:
            print(f"{e}")
            raise
        
    def getInput(self) -> List[List[np.float64]]:
        input = self.__parsedCSV[[self.__input0ColumnName, self.__input1ColumnName]].values
        return input

    def getOutput(self) -> List[List[np.float64]]:
        output = self.__parsedCSV[[self.__output0ColumnName, self.__output1ColumnName]].values
        return output
    
    def _getInput0ColumnName(self) -> str:
        return self.__input0ColumnName
    
    def _getInput1ColumnName(self) -> str:
        return self.__input1ColumnName
    
    def _getOutput0ColumnName(self) -> str:
        return self.__output0ColumnName
    
    def _getOutput1ColumnName(self) -> str:
        return self.__output1ColumnName
    
    def _getCSVFilePath(self) -> str:
        return self.__csv_file_path
    
    def _getParsedCSV(self) -> str:
        return self.__parsedCSV

    def _setCSVFilePath(self, csv_file_path):
        self.__csv_file_path = csv_file_path
        
    def _setCSVFileFinalPath(self) :
        self.__csv_file_path = os.path.join(self.__csv_file_path, 'r2', 'r2_21_100k.csv')
        return;
