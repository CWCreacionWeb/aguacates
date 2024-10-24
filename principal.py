import pandas as pd
import numpy as np

def Cargar(pFile):
    data = pd.read_csv(pFile)
    return data

Datos =Cargar('datos/avocado.csv')

print(len(Datos))

