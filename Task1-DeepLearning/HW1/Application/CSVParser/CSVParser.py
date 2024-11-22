import pandas as pd

class CSVParser():

    def __init__(self, filepath: str, separator: str, headerValue: int): 

        if type(self) is CSVParser:
            raise NotImplementedError("Non è possibile creare un'istanza della classe astratta!")
           
        self.__filepath = filepath
        self.__separator = separator
        self.__headerValue = headerValue

    def __str__(self) -> str:
        
        return f"""CSVParser [filepath={self.__filepath}, separator={self.__separator}, headerValue={self.__headerValue}"""