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
from APPModels.APP_FUN import  APP_Enunciados, chart

# --------------------- 2. Gráficos para Visualización de Datos ---------------------
def P21_Grafico_Violin_Volumen_Venta_Region():
    APP_Enunciados.getEnunciado("2.1")
    chart.figureConfig(title="Distribución de Ventas por Región (Top 5 Regiones)",xlabel="Región",ylabel="Volumen Total de Ventas")
    sns.violinplot(x='region', y='Total Volume', data=chart.df, hue='region', palette="muted", dodge=False, legend=False)

    APP_Enunciados.getExplicacion("2.1")

def P3_2_Elasticidad_Regiones():
    APP_Enunciados.getEnunciado("3.2")
    chart.df['Date'] = pd.to_datetime(chart.df['Date'])
    chart.df['year'] = chart.df['Date'].dt.year

    # Agrupamos por 'region' y 'year' y calculamos el volumen total y precio promedio anual
    df_grouped = chart.df.groupby(['region', 'year']).agg(
        {
            'Total Volume': 'sum', 
            'AveragePrice': 'mean'
        }
    ).reset_index()

    # Calculamos el cambio porcentual por región para 'Total Volume' y 'AveragePrice'
    df_grouped['pct_change_volume'] = df_grouped.groupby('region')['Total Volume'].pct_change()
    df_grouped['pct_change_price'] = df_grouped.groupby('region')['AveragePrice'].pct_change()


    # Calculamos la elasticidad para cada región y año
    df_grouped['elasticity'] = df_grouped['pct_change_volume'] / df_grouped['pct_change_price']

    # Filtramos los valores NaN que pueden haber resultado del cálculo de pct_change en los primeros valores
    df_elasticity = df_grouped.dropna(subset=['elasticity'])

    chart.figureConfig(title="Comparación de Elasticidad en Diferentes Mercados",xlabel="Year",ylabel="Elasticidad")
    for region in df_elasticity['region'].unique():
        region_data = df_elasticity[df_elasticity['region'] == region]
        plt.plot(region_data['year'],region_data['elasticity'],label=region_data['region'])


    plt.show()

def P3_2_Elasticidad_RegionesB():
    # Mostrar enunciado
    APP_Enunciados.getEnunciado("3.2")

    # Convertimos la columna de fechas a datetime y extraemos el año
    chart.df['Date'] = pd.to_datetime(chart.df['Date'])
    chart.df['year'] = chart.df['Date'].dt.year

    # Agrupamos por 'region' y 'year' y calculamos el volumen total y precio promedio anual
    df_grouped = chart.df.groupby(['region', 'year']).agg(
        {
            'Total Volume': 'sum', 
            'AveragePrice': 'mean'
        }
    ).reset_index()

    # Calculamos el cambio porcentual por región para 'Total Volume' y 'AveragePrice'
    df_grouped['pct_change_volume'] = df_grouped.groupby('region')['Total Volume'].pct_change()
    df_grouped['pct_change_price'] = df_grouped.groupby('region')['AveragePrice'].pct_change()

    # Calculamos la elasticidad para cada región y año
    df_grouped['elasticity'] = df_grouped['pct_change_volume'] / df_grouped['pct_change_price']

    # Filtramos los valores NaN que pueden haber resultado del cálculo de pct_change en los primeros valores
    df_elasticity = df_grouped.dropna(subset=['elasticity'])

    # Configuración de la figura
    chart.figureConfig(title="Comparación de Elasticidad en Diferentes Mercados", xlabel="Year", ylabel="Elasticidad")

    # Crear el gráfico de barras
    fig, ax = plt.subplots(figsize=(10, 6))

    # Definimos un ancho de barra para cada región y generamos un desplazamiento entre las barras de cada región
    bar_width = 0.15
    years = df_elasticity['year'].unique()
    x_positions = range(len(years))

    for i, region in enumerate(df_elasticity['region'].unique()):
        region_data = df_elasticity[df_elasticity['region'] == region]
        
        # Usamos un desplazamiento en el eje X para cada región
        ax.bar(
            [x + i * bar_width for x in x_positions],  # Posición en X con desplazamiento
            region_data['elasticity'],                 # Valores de elasticidad
            width=bar_width,                           # Ancho de cada barra
            label=region                               # Etiqueta de la región
        )

    # Configuración de etiquetas y leyenda
    ax.set_xticks([x + (len(df_elasticity['region'].unique()) - 1) * bar_width / 2 for x in x_positions])
    ax.set_xticklabels(years)
    ax.legend(title="Región")

    # Mostrar el gráfico
    plt.show()