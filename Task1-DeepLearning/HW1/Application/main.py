import os
from CSVParser.CSVParser_2J_2D import CSVParser_2J_2D
from CSVParser.CSVParser_3J_2D import CSVParser_3J_2D
from CSVParser.CSVParser_5J_3D import CSVParser_5J_3D

# 1. Ottieni la directory in cui si trova lo script
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Naviga alla cartella superiore (my_project)
parent_dir = os.path.dirname(script_dir)

# 3. Costruisci il percorso completo al file CSV
csv_file_path = os.path.join(parent_dir, 'Dataset', 'Training-set', 'r2', 'r2_21_100k.csv')

parser = CSVParser_2J_2D(csv_file_path, ";", 0)
input = parser.getInput()
output = parser.getOutput()
print(input)
print(output)