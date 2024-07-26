import pandas as pd
import sklearn
import os 

File = "BDCars.csv" 
Filename = os.path.join(os.getcwd(),'Data',File)
print(f'Filename with path: \n {Filename}')