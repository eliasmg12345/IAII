import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from pandas import read_excel
import pandas
import numpy as np
import matplotlib.pyplot as plt
def regresion6():
    # Se lee el archivo excel en un Pandas DataFrame: una tabla
    # con datos mensuales de temperaturas máximas y mínimas,
    # precipitación y horas de sol, de 1957 a 2019
    clima= pd.read_excel('tiempo2.xlsx', sheet_name= 'datos')
    print(clima)
    # Consulta a datos de clima que devuelve filas con mes 7 y
    # crea el marco de datos julio
    julio = clima.query ('Month == 7')
    # Inserta la columna Yr que numera filas de 0 a longitud tabla
    julio.insert (0, 'Yr', range (0, len (julio)))
    # Grafica temperaturas máximas de julio desde 1957
    julio.plot (y = 'Tmax', x = 'Yr')
    # Regresión lineal, polinomio de mínimos cuadrados
    d = np.polyfit (julio ['Yr'], julio ['Tmax'], 1)
    f = np.poly1d (d)
    print(f)
    # Con la función f se produce datos de regresión lineal e
    # inserta en la nueva columna Treg
    julio.insert (6, 'Treg', f (julio ['Yr']))
    # Gráficos de líneas azul Yr contra Tmax y rojo Yr contra Treg
    # (regresión lineal)
    ax = julio.plot (x = 'Yr', y = 'Tmax')
    julio.plot (x = 'Yr', y = 'Treg', color = 'Red', ax = ax)
    # El mismo gráfico, pero de dispersión
    ax = julio.plot.scatter (x = 'Yr', y = 'Tmax')
    julio.plot (x= 'Yr', y= 'Treg', color= 'Red', legend= False, ax= ax)
    plt.show()
