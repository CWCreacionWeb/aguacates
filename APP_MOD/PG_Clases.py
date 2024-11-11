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
    #df_elasticity = df_grouped.dropna(subset=['elasticity'])

    #lo mismo de antes eliminamos el 2015 para que no aparezca año sin datos
    df_elasticity = df_grouped[df_grouped['year'] != 2015]
    #display(df_elasticity)


    # Crear el gráfico de barras
    fig, ax = plt.subplots(figsize=(10, 6))

    # Definimos un ancho de barra para cada región y generamos un desplazamiento entre las barras de cada región
    bar_width = 0.08
    years = df_elasticity['year'].unique()
    x_positions = range(len(years))

    for i, region in enumerate(df_elasticity['region'].unique()):
        region_data = df_elasticity[df_elasticity['region'] == region]
        
        # Usamos un desplazamiento en el eje X para cada región
        bars = ax.bar(
            [x + i * bar_width for x in x_positions],  # Posición en X con desplazamiento
            region_data['elasticity'],                 # Valores de elasticidad
            width=bar_width,                           # Ancho de cada barra
            label=region                               # Etiqueta de la región
        )
        # Añadir las etiquetas encima de cada barra en el formato deseado
        for j, bar in enumerate(bars):
            # Obtener los valores de cambio porcentual y elasticidad
            pct_change_vol = region_data['pct_change_volume'].iloc[j] * 100  # Convertir a porcentaje
            pct_change_prc = region_data['pct_change_price'].iloc[j] * 100  # Convertir a porcentaje
            elasticity = region_data['elasticity'].iloc[j]

            # Formato de la etiqueta: "%cambio volumen / %cambio precio = elasticidad"
            label = f"{pct_change_vol:.0f}%/{pct_change_prc:.0f}%={elasticity:.2f}"

            # Colocar la etiqueta encima de la barra
            ax.text(
                bar.get_x() + bar.get_width() / 2,  # Centrado horizontal de la barra
                #bar.get_height() + 0.05,            # Colocar encima de la barra
                10,                                 # Colocar en el valor Y=0
                label,                              # El texto a mostrar
                ha='center',                        # Alinear al centro
                va='bottom',                        # Alinear al borde inferior (arriba de la barra)
                rotation=90,                        # Rotar la etiqueta para que quede vertical
                fontsize=9                           # Tamaño de la fuente
            )


    # Configuración de etiquetas y leyenda
    #ax.set_yscale('log')

        # Ajustar el límite del eje Y para elasticidad entre -50 y 50
    ax.set_ylim(-50, 50)
    ax.set_xticks([x + (len(df_elasticity['region'].unique()) - 1) * bar_width / 2 for x in x_positions])
    ax.set_xticklabels(years)
    ax.legend(title="Región")

    # Mostrar el gráfico
    plt.show()

def P4_4_CohortesClientesVentas():
    APP_Enunciados.getEnunciado("4.4")
    # Convertir la columna 'Date' a datetime
    chart.df = chart.df[(chart.df['year'] >= 2015) & (chart.df['year'] <= 2017)]
    # Agrupar por region y Date, calculando el total de ventas
    # [pd.Grouper(key='Date', freq='ME'), 'region']
    ventas_por_region_fecha = chart.df.groupby(['Date','region']).agg(
        TotalVolume=('Total Volume', 'sum')
    ).reset_index()

    # Clasificar las regiones en cohortes según el volumen de ventas promedio total
    # Calculamos el volumen promedio para cada región
    volumen_promedio_por_region = ventas_por_region_fecha.groupby('region')['TotalVolume'].mean()

    # Definimos las cohortes: Baja, Media, Alta en función del cuartil
    cuartiles = pd.qcut(volumen_promedio_por_region, 3, labels=['Baja', 'Media', 'Alta'])

    # Agregar la cohorte a cada región en el DataFrame original
    ventas_por_region_fecha['Cohorte'] = ventas_por_region_fecha['region'].map(cuartiles)

    # Crear el gráfico de líneas para cada cohorte
    chart.figureConfig(title="Cohortes según el volumen de ventas por Región",
                       xlabel="Fecha", ylabel="Volumen Total de Ventas")

    for cohorte in ventas_por_region_fecha['Cohorte'].unique():
        subset = ventas_por_region_fecha[ventas_por_region_fecha['Cohorte'] == cohorte]
        subset_grouped = subset.groupby('Date')['TotalVolume'].sum()

        plt.plot(subset_grouped.index, subset_grouped, label=f'Cohorte {cohorte}')

    # chart.plt.savefig("graficas/grafica_4_4.png")
    plt.legend(title='Cohorte de ')
    plt.show()
    APP_Enunciados.getExplicacion("4.4")