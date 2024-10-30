
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime
import numpy as np
import timeit
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import math
import pandas as pd

Datos =''    

# --------------------- 2. Gráficos para Visualización de Datos ---------------------

def P21_Grafico_Violin_Volumen_Venta_Region(pListaRegiones =''):

    plt.figure(figsize=(12, 6))
    if pListaRegiones =='':
        SubData = Datos[Datos['region'] == 'TotalUS']
    else:
        SubData = Datos[Datos['region'].isin(pListaRegiones)]
    sns.violinplot(x='region', y='Total Volume', data=SubData)
    #sns.violinplot(x=Datos['region'],y=Datos['Total Volume'])
    plt.title("Distribución del Volumen Total de Ventas por Región")
    plt.xlabel("Región")
    plt.ylabel("Volumen Total")
    plt.xticks(rotation=90)
    plt.show()

def P22_Boxplot_Comparativo_Precios_Entre_Años():
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='CalYear', y='AveragePrice', data=Datos)
    plt.title("Distribución de Precios Promedios entre Años")
    plt.xlabel("Año")
    plt.ylabel("Precio Promedio")
    plt.show()

def P23_Histograma_Volumen_Total_Ventas():
    plt.figure(figsize=(8, 5))
    plt.hist(Datos['Total Volume'], bins=30, edgecolor='black')
    plt.title("Distribución del Volumen Total de Ventas")
    plt.xlabel("Volumen Total")
    plt.ylabel("Frecuencia")
    plt.show()

def P24_Grafico_Barras_Ventas_Tipo_Bolsa():
    bags = ['Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags']
    total_bags = Datos[bags].sum()
    plt.figure(figsize=(8, 5))
    plt.bar(bags, total_bags, color='skyblue')
    plt.title("Comparación de Ventas por Tipo de Bolsa")
    plt.xlabel("Tipo de Bolsa")
    plt.ylabel("Ventas Totales")
    plt.show()

def P25_Grafico_Lineas_Precios_Promedios_Año():
    avg_price_by_year = Datos.groupby('CalYear')['AveragePrice'].mean()
    plt.figure(figsize=(10, 6))
    plt.plot(avg_price_by_year.index, avg_price_by_year.values, marker='o')
    plt.title("Tendencia de Precios Promedios por Año")
    plt.xlabel("Año")
    plt.ylabel("Precio Promedio")
    plt.show()

