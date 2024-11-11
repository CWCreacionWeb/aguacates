from IPython.display import display, Markdown, HTML
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime
import numpy as np
import timeit
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import math
import pandas as pd
from APPModels.APP_FUN import APP_Enunciados,chart


Datos =''    

# --------------------- 2. Gráficos para Visualización de Datos ---------------------

def P22_Boxplot_Comparativo_Precios_Entre_Años():
    APP_Enunciados.getEnunciado('2.2')

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='CalYear', y='AveragePrice', data=Datos)
    plt.title("Distribución de Precios Promedios entre Años")
    plt.xlabel("Año")
    plt.ylabel("Precio Promedio")
    plt.show()

def P23_Histograma_Volumen_Total_Ventas(pbins=30, pLog=False):
    APP_Enunciados.getEnunciado('2.3')

    mDbg = f"""- **parametros**:  
         - *pbins:*`{pbins}` Numero de intervalos
         - *pLog:*`{pLog}` Escala logaritmica True/False
    """

    display(Markdown(mDbg))

       #if pTipoEscala=='log':
        # Cambiar la escala del eje y a logarítmica
    #plt.yscale('log')

    plt.figure(figsize=(14, 5))

    # Histograma con escala normal
    plt.subplot(1, 2, 1)
    plt.hist(Datos['Total Volume'], bins=pbins, edgecolor='black',log=False)
    plt.title("Distribución del Volumen Total de Ventas")
    plt.xlabel("Volumen Total")
    plt.ylabel("Frecuencia")

    # Histograma con escala logaritmica
    plt.subplot(1, 2, 2)
    plt.hist(Datos['Total Volume'], bins=pbins, edgecolor='black',log=True)
    plt.title("Distribución del Volumen Total de Ventas")
    plt.xlabel("Volumen Total")
    plt.ylabel("Frecuencia (Escala Logarítmica)")
    
    plt.show()
    plt.tight_layout()
    plt.show()    

def P24_Grafico_Barras_Ventas_Tipo_Bolsa():
    APP_Enunciados.getEnunciado('2.4')

    bags = ['Total Bags','Small Bags', 'Large Bags', 'XLarge Bags']
    total_bags = Datos[bags].sum() / 1000000  # Convertir a millones
    plt.figure(figsize=(8, 5))
    #bars=plt.bar(bags, total_bags, color='skyblue')
    bars=plt.bar(bags, total_bags.values, color='skyblue')
    plt.title("Comparación de Ventas por Tipo de Bolsa")
    plt.xlabel("Tipo de Bolsa")
    plt.ylabel("Ventas Totales (millones)")
    # Añadir los valores debajo de cada barra
    for bar, total in zip(bars, total_bags):
        plt.text(
            bar.get_x() + bar.get_width() / 2,   # Posición en X, centrada
            bar.get_height() - (0.05 * bar.get_height()),  # Posición en Y, ligeramente por debajo de la barra
            f'{total:,.0f}',                     # Formato del texto, redondeado sin decimales
            ha='center', va='top', fontsize=10, color='black'
        )

    plt.show()

def P25_Grafico_Lineas_Precios_Promedios_Año(pAnos=''):
    APP_Enunciados.getEnunciado('2.5')

    mDbg = f"""- **parametros**:  
         - *pAnos:*\t`{[pAnos]}`
    """

    display(Markdown(mDbg))

    DatosF = Datos 
    if pAnos =='':
        DatosF = DatosF
    else:
       DatosF = DatosF[DatosF['CalYear'].isin(pAnos)] 



    avg_price_by_year = DatosF.groupby('CalYear')['AveragePrice'].mean()
    plt.figure(figsize=(10, 6))
    plt.plot(avg_price_by_year.index, avg_price_by_year.values, marker='o')
    plt.title("Tendencia de Precios Promedios por Año")
    plt.xlabel("Año")
    plt.ylabel("Precio Promedio")
    plt.show()

