from IPython.display import FileLink
import matplotlib.dates as mdates
from dateutil.relativedelta import relativedelta
from sklearn.ensemble import RandomForestRegressor 
from datetime import datetime
import time
from IPython.display import display, Markdown, HTML,Image

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import statsmodels.api as sm 
from APPModels.APP_FUN import APP_Enunciados,chart
import APPModels.APP_FUN as app_fun  # Importa el módulo completo

from scipy.optimize import minimize
from itertools import combinations
from IPython.display import clear_output, display


import matplotlib.pyplot as plt
from ipywidgets import widgets
from IPython.display import display
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from scipy.optimize import curve_fit

gPre = None
Datos_mensuales =pd.DataFrame()
DatosP103 = pd.DataFrame()
def Inicio():
    global DatosP103
    DatosTmp = app_fun.APP_DatosORG.copy()
    #DatosP103 = DatosTmp[(DatosTmp['region'] == 'TotalUS') & (DatosTmp['type'] == 'conventional')].copy()
    DatosP103 = DatosTmp[(DatosTmp['region'] == 'TotalUS') & (DatosTmp['type'] == 'organic')].copy()
    DatosP103['Cal_AAAAMM_str'] = DatosP103['Cal_AAAADDD'].astype(str)

def calcular_elasticidad(df, columna_precio='AveragePrice', columna_volumen='Total Volume'):
    """
    Calcula la elasticidad precio-demanda entre dos columnas de un DataFrame.
    
    Parámetros:
        df (pd.DataFrame): DataFrame que contiene los datos.
        columna_precio (str): Nombre de la columna de precios.
        columna_volumen (str): Nombre de la columna de volumen.
        
    Retorna:
        pd.Series: Serie con la elasticidad calculada.
    """
    # Calcular los cambios porcentuales de precio y volumen
    cambio_precio = df[columna_precio].pct_change()
    cambio_volumen = df[columna_volumen].pct_change()
    
    # Calcular la elasticidad dividiendo el cambio en volumen entre el cambio en precio
    elasticidad = cambio_volumen / cambio_precio
    
    return elasticidad, cambio_precio, cambio_volumen

def P3_1_Elasticidad_Precio_Demanda_01():
    print('P3_1_Elasticidad_Precio_Demanda_01')
    datos = DatosP103.copy()
    datos = datos.sort_values(by='Date', ascending=True)

    # Calcular elasticidad diaria
    datos['Elasticidad'],datos['CambioPrecio'], datos['CambioVolumen']   = calcular_elasticidad(datos)
    

    # Alinear y asignar la columna usando índices explícitamente
    DatosP103.loc[datos.index, 'Elasticidad'] = datos['Elasticidad']
    DatosP103.loc[datos.index, 'CambioPrecio'] = datos['CambioPrecio']
    DatosP103.loc[datos.index, 'CambioVolumen'] = datos['CambioVolumen']


    Data01_MediaMensual_ElasticidadesSemanales(datos)

        # Agrupar los datos por región y aplicar el cálculo de elasticidad
    #datos[['Elasticidad', 'CambioPrecio', 'CambioVolumen']] = datos.groupby('region').apply(
    #    lambda x: pd.Series(calcular_elasticidad(x)))

    # Definir el rango del eje x en función de 'meses_mostrar'
    vfecha_inicio = datos['CalFecha'].min()
    vfecha_fin = datos['CalFecha'].max()
    #vfecha_fin = vfecha_inicio + relativedelta(months=3)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 100)
    datos = datos.sort_values(by='CalFecha')
    # Exportar a un archivo Excel
    archivo_salida = 'P103_elasticidad_precio_demanda.xlsx'
    datos.to_excel(archivo_salida, index=False)  # Exporta el DataFrame a un archivo Excel
    display(FileLink(archivo_salida))


    fig, ax = plt.subplots(figsize=(14, 8))  # Usamos subplots para obtener tanto la figura como el eje


    datos.plot(x='CalFecha', y='Elasticidad', ax=ax, title='Elasticidad Precio-Demanda por Día')

    vFecha = datos.iloc[1]['CalFecha']
    # Configurar el gráfico
    ax.tick_params(axis='x', rotation=45)
    #ax.xaxis.set_major_locator(mdates.MonthLocator())
    ##ax.xaxis.set_major_formatter(mdates.DateFormatter(f'%Y-%m'))    
    ax.set_ylim(-10, +10)  # Limitar el rango del eje y entre -10 y 10
    #ax.set_xlim(vfecha_inicio, vfecha_fin)  # Limitar el rango del eje x a los meses indicados

    ax.grid(True, axis='y', linestyle='--', color='gray', alpha=0.7)  # Líneas horizontales (en y) con estilo
    plt.show()    


def P3_1_Elasticidad_Precio_Demanda_02A(agrupacion='MM'):
    print('P3_1_Elasticidad_Precio_Demanda_02')
    global DatosP103
    datos = DatosP103.copy()
    datos['Cal_AAAAMM_str'] = datos['Cal_AAAAMM'].astype(str)
    df_agg = None
    
    if agrupacion == 'MM':
        df_agg = datos.groupby('Cal_AAAAMM_str').agg({
            'AveragePrice': 'mean',  
            'Total Volume': 'sum'
        }).reset_index()
    elif agrupacion == 'AAAA':
        df_agg = datos.groupby('Cal_AAAA').agg({
            'AveragePrice': 'mean',  
            'Total Volume': 'sum'
        }).reset_index()
    elif agrupacion == 'TOTAL':
        df_agg = datos.mean().to_frame().T 
        
    # Aplicar la función de elasticidad
    df_agg['Elasticidad'], df_agg['CambioPrecio'], df_agg['CambioVolumen'] = calcular_elasticidad(df_agg)

    archivo_salida = 'P103_elasticidad_precio_demanda_02.xlsx'
    df_agg.to_excel(archivo_salida, index=False)

    # Configurar el gráfico de barras
    fig, ax = plt.subplots(figsize=(15, 8))  # Tamaño adaptado para un ajuste cómodo en JupyterLab
    
    # Graficar elasticidad agrupada como gráfico de barras
    ax = df_agg.plot(
        x='Cal_AAAAMM_str',
        y='Elasticidad',
        kind='bar',
        ax=ax,
        title=f'Elasticidad Precio-Demanda Agrupada ({agrupacion})'
    )
    
    # Etiquetas y rotación
    ax.set_xticks(df_agg.index)
    ax.set_xticklabels(df_agg['Cal_AAAAMM_str'], rotation=45, ha='right')
    plt.xlabel("Fecha")
    plt.ylabel("Elasticidad")
    ax.grid(True, linestyle='--', color='gray', alpha=0.7)

    # Configurar tamaño inicial del gráfico (mitad de datos)
    half_length = len(df_agg) // 2
    ax.set_xlim(0, half_length)

    # Configuración del control deslizante
    scroll = widgets.IntSlider(
        min=0,
        max=len(df_agg) - half_length,
        step=1,
        description="Scroll:"
    )
    display(scroll)

    # Función para actualizar el gráfico en función del scroll
    def on_scroll(change):
        start = change['new']
        ax.set_xlim(start, start + half_length)
        fig.canvas.draw_idle()
        print('OnScroll')

    # Vincular el control deslizante a la función de desplazamiento
    scroll.observe(on_scroll, names='value')
    plt.show()


def P3_1_Elasticidad_Precio_Demanda_02Aerr( agrupacion='MM'):
    print('P3_1_Elasticidad_Precio_Demanda_02')
    global DatosP103
    datos = DatosP103.copy()
    datos['Cal_AAAAMM_str'] = datos['Cal_AAAAMM'].astype(str)
    df_agg = None
    if agrupacion == 'MM':
        df_agg = datos.groupby('Cal_AAAAMM_str').agg({
                'AveragePrice': 'mean',  # Media para AveragePrice
                'Total Volume': 'sum'    # Suma para Total Volume
            }).reset_index()
    elif agrupacion == 'AAAA':
        df_agg = datos.groupby('Cal_AAAA').agg({
            'AveragePrice': 'mean',  # Media para AveragePrice
            'Total Volume': 'sum'    # Suma para Total Volume
        }).reset_index()
    elif agrupacion == 'TOTAL':
        df_agg = datos.mean().to_frame().T  # DataFrame con un solo registro
        
    # Aplicar la función de elasticidad al DataFrame agregado
    df_agg['Elasticidad'],df_agg['CambioPrecio'], df_agg['CambioVolumen']= calcular_elasticidad(df_agg)

    archivo_salida = 'P103_elasticidad_precio_demanda_02.xlsx'
    df_agg.to_excel(archivo_salida, index=False)  # Exporta el DataFrame a un archivo Excel

      # Crear el gráfico con tamaño grande
    fig, ax = plt.subplots(figsize=(30, 10))  # Ajusta figsize para tamaño de pantalla completo

    # Graficar elasticidad agrupada
    #ax = df_agg.plot(x='Cal_AAAAMM_str',y='Elasticidad', ax=ax, title=f'Elasticidad Precio-Demanda Agrupada ({agrupacion})')   

    # Graficar elasticidad agrupada como gráfico de barras
    ax = df_agg.plot(
        x='Cal_AAAAMM_str',
        y='Elasticidad',
        kind='bar',  # Cambiar a gráfico de barras
        ax=ax,
        title=f'Elasticidad Precio-Demanda Agrupada ({agrupacion})'
    )
    

        # Formatear fechas en AAAA-MM y rotar etiquetas 45 grados
    #ax.set_xticks(df_agg.index[::max(1, len(df_agg) // 14)])  # Mostrar solo 14 fechas
    #ax.set_xticklabels(df_agg.index[::max(1, len(df_agg) // 14)].strftime('%Y-%m'), rotation=45)
    
    # Mostrar todas las etiquetas del eje X
    ax.set_xticks(df_agg.index)  # Establecer todas las fechas como etiquetas del eje X
    ax.set_xticklabels(df_agg['Cal_AAAAMM_str'], rotation=45, ha='right')  # Rotar etiquetas 45 grados para legibilidad
    plt.xlabel("Fecha")
    plt.ylabel("Elasticidad")

    # Añadir las líneas de la cuadrícula (horizontales y verticales)
    ax.grid(True, linestyle='--', color='gray', alpha=0.7)

    # Tamaño de la mitad del gráfico
    half_length = len(df_agg) // 2

    # Configuración de la barra de desplazamiento
    scroll = widgets.IntSlider(
        min=0,
        max=len(df_agg) - half_length,
        step=1,
        description="Scroll:"
    )
    display(scroll)

    """
   # Ajuste con barra de scroll
    scroll = widgets.HBox([widgets.Label(value="Scroll:"), widgets.IntSlider(min=0, max=len(df_agg), step=1)])
    display(scroll)
    """
       # Mostrar inicialmente solo la mitad
    ax.set_xlim(0, half_length - 1)
    plt.show()

    # Función para actualizar el gráfico en función del scroll
    def on_scroll(change):
        start = change['new']
        ax.set_xlim(start, start + half_length - 1)
        fig.canvas.draw_idle()
        print('OnScroll 02')

    # Vincular el evento de cambio en el control deslizante
    scroll.observe(on_scroll, names='value')


def P3_1_Elasticidad_Precio_Demanda_02(agrupacion='MM'):
    print('P3_1_Elasticidad_Precio_Demanda_02')
    global DatosP103
    datos = DatosP103.copy()
    datos['Cal_AAAAMM_str'] = datos['Cal_AAAAMM'].astype(str)
    df_agg = None
    
    if agrupacion == 'MM':
        df_agg = datos.groupby('Cal_AAAAMM_str').agg({
            'AveragePrice': 'mean',  
            'Total Volume': 'sum'
        }).reset_index()
    elif agrupacion == 'AAAA':
        df_agg = datos.groupby('Cal_AAAA').agg({
            'AveragePrice': 'mean',  
            'Total Volume': 'sum'
        }).reset_index()
    elif agrupacion == 'TOTAL':
        df_agg = datos.mean().to_frame().T 
        
    # Aplicar la función de elasticidad
    df_agg['Elasticidad'], df_agg['CambioPrecio'], df_agg['CambioVolumen'] = calcular_elasticidad(df_agg)

    archivo_salida = 'P103_elasticidad_precio_demanda_02.xlsx'
    df_agg.to_excel(archivo_salida, index=False)

    # Crear el gráfico interactivo con Plotly
    fig = px.bar(
        df_agg,
        x='Cal_AAAAMM_str',
        y='Elasticidad',
        title=f'Elasticidad Precio-Demanda Agrupada ({agrupacion})',
        labels={'Cal_AAAAMM_str': 'Fecha', 'Elasticidad': 'Elasticidad'},
        template='plotly_white',
    )

    # Ajustar el diseño para que sea desplazable
    fig.update_layout(
        xaxis=dict(
            tickangle=45,
            rangeslider=dict(visible=True),  # Habilitar la barra de desplazamiento
        ),
        height=600,  # Ajustar el tamaño del gráfico
    )

    # Mostrar el gráfico
    fig.show()


def P3_1_Elasticidad_Precio_Demanda_02_mean(agrupacion='MM'):
    print('P3_1_Elasticidad_Precio_Demanda_02')
    global DatosP103
    datos = DatosP103.copy()
    datos['Cal_AAAAMM_str'] = datos['Cal_AAAAMM'].astype(str)
    df_agg = None
    
    if agrupacion == 'MM':
        df_agg = datos.groupby('Cal_AAAAMM_str').agg({
            'AveragePrice': 'mean',  
            'Total Volume': 'mean'
        }).reset_index()
    elif agrupacion == 'AAAA':
        df_agg = datos.groupby('Cal_AAAA').agg({
            'AveragePrice': 'mean',  
            'Total Volume': 'mean'
        }).reset_index()
    elif agrupacion == 'TOTAL':
        df_agg = datos.mean().to_frame().T 
        
    # Aplicar la función de elasticidad
    df_agg['Elasticidad'], df_agg['CambioPrecio'], df_agg['CambioVolumen'] = calcular_elasticidad(df_agg)

    archivo_salida = 'P103_elasticidad_precio_demanda_02_mean.xlsx'
    df_agg.to_excel(archivo_salida, index=False)

    # Crear el gráfico interactivo con Plotly
    fig = px.bar(
        df_agg,
        x='Cal_AAAAMM_str',
        y='Elasticidad',
        title=f'Elasticidad Precio-Demanda Agrupada ({agrupacion})',
        labels={'Cal_AAAAMM_str': 'Fecha', 'Elasticidad': 'Elasticidad'},
        template='plotly_white',
    )

    # Ajustar el diseño para que sea desplazable
    fig.update_layout(
        xaxis=dict(
            tickangle=45,
            rangeslider=dict(visible=True),  # Habilitar la barra de desplazamiento
        ),
        height=600,  # Ajustar el tamaño del gráfico
    )

    # Mostrar el gráfico
    fig.show()


# Nueva función para mostrar el gráfico de Total Volume (media y suma)
def mostrar_volumen_total_express(agrupacion='MM'):
    print('Mostrar Volumen Total Agrupado por:', agrupacion)
    global DatosP103
    datos = DatosP103.copy()
    datos['Cal_AAAAMM_str'] = datos['Cal_AAAAMM'].astype(str)
    
    # Agrupar los datos por 'Cal_AAAAMM_str' si es MM
    if agrupacion == 'MM':
        df_agg = datos.groupby('Cal_AAAAMM_str').agg({
            'Total Volume': ['mean', 'sum']  # Media y suma de Total Volume
        }).reset_index()
        # Plan de columnas para que se vea más limpio en el gráfico
        df_agg.columns = ['Cal_AAAAMM_str', 'Total Volume Media', 'Total Volume Suma']
    else:
        print("No se ha definido un agrupamiento correcto para esta opción.")
        return
    
    # Crear el gráfico interactivo con Plotly (Media y Suma de Total Volume)
    fig = px.bar(
        df_agg,
        x='Cal_AAAAMM_str',
        y=['Total Volume Media', 'Total Volume Suma'],
        title=f'Media y Suma de Total Volume Agrupada ({agrupacion})',
        labels={'Cal_AAAAMM_str': 'Fecha', 'value': 'Total Volume', 'variable': 'Estadística'},
        template='plotly_white',
    )

    # Ajustar el diseño para el gráfico
    fig.update_layout(
        xaxis=dict(
            tickangle=45,
            rangeslider=dict(visible=True),  # Habilitar la barra de desplazamiento
        ),
        height=600,  # Ajustar el tamaño del gráfico
    )

    # Mostrar el gráfico
    fig.show()

def mostrar_volumen_total_paralelas_express(agrupacion='MM'):
    print('Mostrar Volumen Total Agrupado por:', agrupacion)
    global DatosP103
    datos = DatosP103.copy()
    datos['Cal_AAAAMM_str'] = datos['Cal_AAAAMM'].astype(str)
    
    # Agrupar los datos por 'Cal_AAAAMM_str' si es MM
    if agrupacion == 'MM':
        df_agg = datos.groupby('Cal_AAAAMM_str').agg({
            'Total Volume': ['mean', 'sum']  # Media y suma de Total Volume
        }).reset_index()
        # Plan de columnas para que se vea más limpio en el gráfico
        df_agg.columns = ['Cal_AAAAMM_str', 'Total Volume Media', 'Total Volume Suma']
    else:
        print("No se ha definido un agrupamiento correcto para esta opción.")
        return
    
    # Crear el gráfico interactivo con Plotly (Media y Suma de Total Volume)
    fig = px.bar(
        df_agg,
        x='Cal_AAAAMM_str',
        y=['Total Volume Media', 'Total Volume Suma'],
        title=f'Media y Suma de Total Volume Agrupada ({agrupacion})',
        labels={'Cal_AAAAMM_str': 'Fecha', 'value': 'Total Volume', 'variable': 'Estadística'},
        template='plotly_white',
    )

    # Ajustar el diseño para que las barras aparezcan una al lado de la otra
    fig.update_layout(
        barmode='group',  # Esto agrupa las barras una al lado de la otra
        xaxis=dict(
            tickangle=45,
            title='Fecha'
        ),
        yaxis=dict(
            title='Total Volume'
        ),
        height=600,  # Ajustar el tamaño del gráfico
    )

    # Mostrar el gráfico
    fig.show()


# Nueva función para mostrar el gráfico de Total Volume (Media y Suma) con dos escalas
def mostrar_volumen_total_paralelas02_express(agrupacion='MM'):
    print('Mostrar Volumen Total Agrupado por:', agrupacion)
    global DatosP103
    datos = DatosP103.copy()
    datos['Cal_AAAAMM_str'] = datos['Cal_AAAAMM'].astype(str)
    
    # Agrupar los datos por 'Cal_AAAAMM_str' si es MM
    if agrupacion == 'MM':
        df_agg = datos.groupby('Cal_AAAAMM_str').agg({
            'Total Volume': ['mean', 'sum']  # Media y suma de Total Volume
        }).reset_index()
        # Plan de columnas para que se vea más limpio en el gráfico
        df_agg.columns = ['Cal_AAAAMM_str', 'Total Volume Media', 'Total Volume Suma']
    else:
        print("No se ha definido un agrupamiento correcto para esta opción.")
        return
    
    # Crear el gráfico interactivo con Plotly (Media y Suma de Total Volume)
    fig = go.Figure()

    # Agregar la barra de Total Volume Media en el eje Y izquierdo
    fig.add_trace(go.Bar(
        x=df_agg['Cal_AAAAMM_str'],
        y=df_agg['Total Volume Media'],
        name='Total Volume Media',
        yaxis='y1'  # Asignamos al eje Y izquierdo
    ))

    # Agregar la barra de Total Volume Suma en el eje Y derecho
    fig.add_trace(go.Bar(
        x=df_agg['Cal_AAAAMM_str'],
        y=df_agg['Total Volume Suma'],
        name='Total Volume Suma',
        yaxis='y2'  # Asignamos al eje Y derecho
    ))

    # Ajustar el diseño para que las barras aparezcan una al lado de la otra
    fig.update_layout(
        barmode='group',  # Esto agrupa las barras una al lado de la otra
        title=f'Media y Suma de Total Volume Agrupada ({agrupacion})',
        xaxis=dict(
            tickangle=45,
            title='Fecha'
        ),
        yaxis=dict(
            title='Total Volume Media',  # Título para el eje izquierdo
        ),
        yaxis2=dict(
            title='Total Volume Suma',  # Título para el eje derecho
            overlaying='y',  # Superponer con el eje Y izquierdo
            side='right',  # Colocar el eje en el lado derecho
        ),
        height=600,  # Ajustar el tamaño del gráfico
    )

    # Mostrar el gráfico
    fig.show()

# Nueva función para mostrar el gráfico de Total Volume (Media y Suma) con dos escalas
def mostrar_volumen_total_paralelas03_express(agrupacion='MM'):
    print('Mostrar Volumen Total Agrupado por:', agrupacion)
    global DatosP103
    datos = DatosP103.copy()
    datos['Cal_AAAAMM_str'] = datos['Cal_AAAAMM'].astype(str)
    
    # Agrupar los datos por 'Cal_AAAAMM_str' si es MM
    if agrupacion == 'MM':
        df_agg = datos.groupby('Cal_AAAAMM_str').agg({
            'Total Volume': ['mean', 'sum']  # Media y suma de Total Volume
        }).reset_index()
        # Plan de columnas para que se vea más limpio en el gráfico
        df_agg.columns = ['Cal_AAAAMM_str', 'Total Volume Media', 'Total Volume Suma']
    else:
        print("No se ha definido un agrupamiento correcto para esta opción.")
        return
    
    # Crear el gráfico interactivo con Plotly (Media y Suma de Total Volume)
    fig = go.Figure()

    # Agregar la barra de Total Volume Media en el eje Y izquierdo
    fig.add_trace(go.Bar(
        x=df_agg['Cal_AAAAMM_str'],
        y=df_agg['Total Volume Media'],
        name='Total Volume Media',
        yaxis='y1',  # Asignamos al eje Y izquierdo
        width=0.4,  # Establecemos el ancho de las barras para que estén separadas
    ))

    # Agregar la barra de Total Volume Suma en el eje Y derecho
    fig.add_trace(go.Bar(
        x=df_agg['Cal_AAAAMM_str'],
        y=df_agg['Total Volume Suma'],
        name='Total Volume Suma',
        yaxis='y2',  # Asignamos al eje Y derecho
        width=0.4,  # Establecemos el ancho de las barras para que estén separadas
    ))

    # Ajustar el diseño para que las barras aparezcan una al lado de la otra
    fig.update_layout(
        barmode='group',  # Esto agrupa las barras una al lado de la otra
        title=f'Media y Suma de Total Volume Agrupada ({agrupacion})',
        xaxis=dict(
            tickangle=45,
            title='Fecha'
        ),
        yaxis=dict(
            title='Total Volume Media',  # Título para el eje izquierdo
        ),
        yaxis2=dict(
            title='Total Volume Suma',  # Título para el eje derecho
            overlaying='y',  # Superponer con el eje Y izquierdo
            side='right',  # Colocar el eje en el lado derecho
        ),
        height=600,  # Ajustar el tamaño del gráfico
    )

    # Mostrar el gráfico
    fig.show()


# Nueva función para mostrar el gráfico de Total Volume (Media y Suma) con dos escalas
def mostrar_volumen_total_paralelas04_express(agrupacion='MM'):
    print('Mostrar Volumen Total Agrupado por:', agrupacion)
    global DatosP103
    datos = DatosP103.copy()
    datos['Cal_AAAAMM_str'] = datos['Cal_AAAAMM'].astype(str)
    
    # Agrupar los datos por 'Cal_AAAAMM_str' si es MM
    if agrupacion == 'MM':
        df_agg = datos.groupby('Cal_AAAAMM_str').agg({
            'Total Volume': ['mean', 'sum']  # Media y suma de Total Volume
        }).reset_index()
        # Plan de columnas para que se vea más limpio en el gráfico
        df_agg.columns = ['Cal_AAAAMM_str', 'Total Volume Media', 'Total Volume Suma']
    else:
        print("No se ha definido un agrupamiento correcto para esta opción.")
        return
    
    # Crear el gráfico interactivo con Plotly (Media y Suma de Total Volume)
    fig = go.Figure()

    # Desplazamiento para las barras para que no se superpongan
    offset = 0.2  # Ajuste del desplazamiento de las barras

    # Agregar la barra de Total Volume Media en el eje Y izquierdo
    fig.add_trace(go.Bar(
        x=df_agg['Cal_AAAAMM_str'],
        y=df_agg['Total Volume Media'],
        name='Total Volume Media',
        yaxis='y1',  # Asignamos al eje Y izquierdo
        width=0.4,  # Establecemos el ancho de las barras para que estén separadas
        offsetgroup=0  # Desplazamos las barras de este grupo
    ))

    # Agregar la barra de Total Volume Suma en el eje Y derecho
    fig.add_trace(go.Bar(
        x=df_agg['Cal_AAAAMM_str'],
        y=df_agg['Total Volume Suma'],
        name='Total Volume Suma',
        yaxis='y2',  # Asignamos al eje Y derecho
        width=0.4,  # Establecemos el ancho de las barras para que estén separadas
        offsetgroup=1  # Desplazamos las barras de este grupo
    ))

    # Ajustar el diseño para que las barras aparezcan una al lado de la otra
    fig.update_layout(
        barmode='group',  # Esto agrupa las barras una al lado de la otra
        title=f'Media y Suma de Total Volume Agrupada ({agrupacion})',
        xaxis=dict(
            tickangle=45,
            title='Fecha'
        ),
        yaxis=dict(
            title='Total Volume Media',  # Título para el eje izquierdo
        ),
        yaxis2=dict(
            title='Total Volume Suma',  # Título para el eje derecho
            overlaying='y',  # Superponer con el eje Y izquierdo
            side='right',  # Colocar el eje en el lado derecho
        ),
        height=600,  # Ajustar el tamaño del gráfico
    )

    # Mostrar el gráfico
    fig.show()

# Nueva función para mostrar el gráfico de Total Volume (Media y Suma) con una o dos escalas
def mostrar_volumen_total_paralelas05_express(agrupacion='MM', dos_escalas=False):
    print('Mostrar Volumen Total Agrupado por:', agrupacion)
    global DatosP103
    datos = DatosP103.copy()
    datos['Cal_AAAAMM_str'] = datos['Cal_AAAAMM'].astype(str)
    
    # Agrupar los datos por 'Cal_AAAAMM_str' si es MM
    if agrupacion == 'MM':
        df_agg = datos.groupby('Cal_AAAAMM_str').agg({
            'Total Volume': ['mean', 'sum']  # Media y suma de Total Volume
        }).reset_index()
        # Plan de columnas para que se vea más limpio en el gráfico
        df_agg.columns = ['Cal_AAAAMM_str', 'Total Volume Media', 'Total Volume Suma']
    else:
        print("No se ha definido un agrupamiento correcto para esta opción.")
        return
    
    # Crear el gráfico interactivo con Plotly (Media y Suma de Total Volume)
    fig = go.Figure()

    # Desplazamiento para las barras para que no se superpongan
    offset = 0.2  # Ajuste del desplazamiento de las barras

    if dos_escalas:
        # Usamos dos escalas (izquierda para Media y derecha para Suma)
        fig.add_trace(go.Bar(
            x=df_agg['Cal_AAAAMM_str'],
            y=df_agg['Total Volume Media'],
            name='Total Volume Media',
            yaxis='y1',  # Asignamos al eje Y izquierdo
            width=0.4,  # Establecemos el ancho de las barras para que estén separadas
            offsetgroup=0  # Desplazamos las barras de este grupo
        ))

        fig.add_trace(go.Bar(
            x=df_agg['Cal_AAAAMM_str'],
            y=df_agg['Total Volume Suma'],
            name='Total Volume Suma',
            yaxis='y2',  # Asignamos al eje Y derecho
            width=0.4,  # Establecemos el ancho de las barras para que estén separadas
            offsetgroup=1  # Desplazamos las barras de este grupo
        ))

        # Ajustar el diseño para que las barras aparezcan una al lado de la otra
        fig.update_layout(
            barmode='group',  # Esto agrupa las barras una al lado de la otra
            title=f'Media y Suma de Total Volume Agrupada ({agrupacion})',
            xaxis=dict(
                tickangle=45,
                title='Fecha'
            ),
            yaxis=dict(
                title='Total Volume Media'  # Título para el eje izquierdo
            ),
            yaxis2=dict(
                title='Total Volume Suma',  # Título para el eje derecho
                overlaying='y',  # Lo superpone al eje Y izquierdo
                side='right',  # Coloca la escala a la derecha
            ),
            height=600,  # Ajustar el tamaño del gráfico
        )

    else:
        # Usamos una sola escala para ambas barras
        fig.add_trace(go.Bar(
            x=df_agg['Cal_AAAAMM_str'],
            y=df_agg['Total Volume Media'],
            name='Total Volume Media',
            yaxis='y1',  # Asignamos al eje Y izquierdo
            width=0.4,  # Establecemos el ancho de las barras para que estén separadas
            offsetgroup=0  # Desplazamos las barras de este grupo
        ))

        fig.add_trace(go.Bar(
            x=df_agg['Cal_AAAAMM_str'],
            y=df_agg['Total Volume Suma'],
            name='Total Volume Suma',
            yaxis='y1',  # Asignamos al mismo eje Y izquierdo
            width=0.4,  # Establecemos el ancho de las barras para que estén separadas
            offsetgroup=1  # Desplazamos las barras de este grupo
        ))

        # Ajustar el diseño para que las barras aparezcan una al lado de la otra
        fig.update_layout(
            barmode='group',  # Esto agrupa las barras una al lado de la otra
            title=f'Media y Suma de Total Volume Agrupada ({agrupacion})',
            xaxis=dict(
                tickangle=45,
                title='Fecha'
            ),
            yaxis=dict(
                title='Total Volume',  # Título para el eje izquierdo
            ),
            height=600,  # Ajustar el tamaño del gráfico
        )

    # Mostrar el gráfico
    fig.show()    

# Nueva función para mostrar el gráfico de Total Volume (Media y Suma) como líneas
def mostrar_volumen_total_paralelas05_lineas(agrupacion='MM', dos_escalas=False):
    print('Mostrar Volumen Total Agrupado por:', agrupacion)
    global DatosP103
    datos = DatosP103.copy()
    
    
    # Agrupar los datos por 'Cal_AAAAMM_str' si es MM
    if agrupacion == 'MM':
        df_agg = datos.groupby('Cal_AAAAMM_str').agg({
            'Total Volume': ['mean', 'sum']  # Media y suma de Total Volume
        }).reset_index()
        # Plan de columnas para que se vea más limpio en el gráfico
        df_agg.columns = ['Cal_AAAAMM_str', 'Total Volume Media', 'Total Volume Suma']
    else:
        print("No se ha definido un agrupamiento correcto para esta opción.")
        return
    
    # Crear el gráfico interactivo con Plotly (Media y Suma de Total Volume)
    fig = go.Figure()

    # Desplazamiento para las líneas (ajustado por separado, ya que no es necesario como en las barras)
    offset = 0.2  # Ajuste de desplazamiento si es necesario, aunque para líneas no es crítico

    if dos_escalas:
        # Usamos dos escalas (izquierda para Media y derecha para Suma)
        fig.add_trace(go.Scatter(
            x=df_agg['Cal_AAAAMM_str'],
            y=df_agg['Total Volume Media'],
            name='Total Volume Media',
            mode='lines+markers',  # Modo línea con marcadores
            yaxis='y1',  # Asignamos al eje Y izquierdo
            line=dict(width=2),  # Ancho de la línea
        ))

        fig.add_trace(go.Scatter(
            x=df_agg['Cal_AAAAMM_str'],
            y=df_agg['Total Volume Suma'],
            name='Total Volume Suma',
            mode='lines+markers',  # Modo línea con marcadores
            yaxis='y2',  # Asignamos al eje Y derecho
            line=dict(width=2),  # Ancho de la línea
        ))

        # Ajustar el diseño para que las líneas aparezcan con dos escalas
        fig.update_layout(
            title=f'Media y Suma de Total Volume Agrupada ({agrupacion})',
            xaxis=dict(
                tickangle=45,
                title='Fecha'
            ),
            yaxis=dict(
                title='Total Volume Media'  # Título para el eje izquierdo
            ),
            yaxis2=dict(
                title='Total Volume Suma',  # Título para el eje derecho
                overlaying='y',  # Lo superpone al eje Y izquierdo
                side='right',  # Coloca la escala a la derecha
            ),
            height=600,  # Ajustar el tamaño del gráfico
        )

    else:
        # Usamos una sola escala para ambas líneas
        fig.add_trace(go.Scatter(
            x=df_agg['Cal_AAAAMM_str'],
            y=df_agg['Total Volume Media'],
            name='Total Volume Media',
            mode='lines+markers',  # Modo línea con marcadores
            yaxis='y1',  # Asignamos al eje Y izquierdo
            line=dict(width=2),  # Ancho de la línea
        ))

        fig.add_trace(go.Scatter(
            x=df_agg['Cal_AAAAMM_str'],
            y=df_agg['Total Volume Suma'],
            name='Total Volume Suma',
            mode='lines+markers',  # Modo línea con marcadores
            yaxis='y1',  # Asignamos al mismo eje Y izquierdo
            line=dict(width=2),  # Ancho de la línea
        ))

        # Ajustar el diseño para que las líneas aparezcan con una sola escala
        fig.update_layout(
            title=f'Media y Suma de Total Volume Agrupada ({agrupacion})',
            xaxis=dict(
                tickangle=45,
                title='Fecha'
            ),
            yaxis=dict(
                title='Total Volume',  # Título para el eje izquierdo
            ),
            height=600,  # Ajustar el tamaño del gráfico
        )

    # Mostrar el gráfico
    fig.show()


# Función para mostrar los gráficos de media y suma de Total Volume
def mostrar_volumen_total(agrupacion='MM'):
    print('Mostrar Volumen Total Agrupado por:', agrupacion)
    global DatosP103
    datos = DatosP103.copy()
    datos['Cal_AAAAMM_str'] = datos['Cal_AAAAMM'].astype(str)
    
    # Agrupar los datos por 'Cal_AAAAMM_str' si es MM
    if agrupacion == 'MM':
        df_agg = datos.groupby('Cal_AAAAMM_str').agg({
            'Total Volume': ['mean', 'sum']  # Media y suma de Total Volume
        }).reset_index()
        # Plan de columnas para que se vea más limpio en el gráfico
        df_agg.columns = ['Cal_AAAAMM_str', 'Total Volume Media', 'Total Volume Suma']
    else:
        print("No se ha definido un agrupamiento correcto para esta opción.")
        return
    
    # Crear el gráfico de barras con dos ejes Y
    fig = make_subplots(
        rows=1, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.1,
        subplot_titles=[f'Media y Suma de Total Volume Agrupada ({agrupacion})'],
        specs=[[{"secondary_y": True}]]  # Esto permite tener un segundo eje Y
    )

    # Agregar la barra para la media de Total Volume
    fig.add_trace(
        go.Bar(
            x=df_agg['Cal_AAAAMM_str'], 
            y=df_agg['Total Volume Media'], 
            name='Total Volume Media',
            marker=dict(color='blue')
        ), 
        secondary_y=False  # Asignamos al eje Y de la izquierda
    )

    # Agregar la barra para la suma de Total Volume
    fig.add_trace(
        go.Bar(
            x=df_agg['Cal_AAAAMM_str'], 
            y=df_agg['Total Volume Suma'], 
            name='Total Volume Suma',
            marker=dict(color='orange')
        ), 
        secondary_y=True  # Asignamos al eje Y de la derecha
    )

    # Actualizar la disposición del gráfico
    fig.update_layout(
        title=f'Media y Suma de Total Volume Agrupada ({agrupacion})',
        barmode='group',  # Esto agrupa las barras lado a lado
        xaxis=dict(
            tickangle=45,  # Rotar las etiquetas del eje X para mejor visibilidad
            title='Fecha'
        ),
        yaxis=dict(
            title='Total Volume Media',  # Título del eje Y izquierdo
            titlefont=dict(color='blue'),
            tickfont=dict(color='blue')
        ),
        yaxis2=dict(
            title='Total Volume Suma',  # Título del eje Y derecho
            titlefont=dict(color='orange'),
            tickfont=dict(color='orange'),
            overlaying='y',  # Superpone este eje al principal
            side='right'  # Lo coloca en el lado derecho
        ),
        height=600,  # Ajustar el tamaño del gráfico
    )

    # Mostrar el gráfico
    fig.show()


    # Nueva función para mostrar la Elasticidad Precio-Demanda para Total Volume Media y Suma
def mostrar_elasticidad_precio_demanda(agrupacion='MM', dos_escalas=False):
    print('Mostrar Elasticidad Precio-Demanda Agrupada por:', agrupacion)

    display(Image(filename='P103IMG/5Semanas.png'))

    global DatosP103
    datos = DatosP103.copy()
    
    # Agrupar los datos por 'Cal_AAAAMM_str' si es MM
    if agrupacion == 'MM':
        if 'mean' == 'mean':
            df_agg = datos.groupby('Cal_AAAAMM_str').agg({
                'Total Volume': ['mean', 'sum'],  # Media y suma de Total Volume
                'AveragePrice': 'mean'  # Media de AveragePrice (precio)
            }).reset_index()
        else:
            df_agg = datos.groupby('Cal_AAAAMM_str').agg({
                'Total Volume': ['median', 'mean'],  # Media y suma de Total Volume
                'AveragePrice': 'median'  # Media de AveragePrice (precio)
            }).reset_index()
        # Plan de columnas para que se vea más limpio en el gráfico
        df_agg.columns = ['Cal_AAAAMM_str', 'Total Volume Media', 'Total Volume Suma', 'AveragePrice']
    else:
        print("No se ha definido un agrupamiento correcto para esta opción.")
        return
    
    # Calcular la Elasticidad Precio-Demanda para Media y Suma
    df_agg['Elasticidad Media'] = calcular_elasticidadS(df_agg['Total Volume Media'], df_agg['AveragePrice'])
    df_agg['Elasticidad Suma'] = calcular_elasticidadS(df_agg['Total Volume Suma'], df_agg['AveragePrice'])
    
    Data01_MediaMensual_ADD(df_agg, 'Elasticidad Media')
    Data01_MediaMensual_ADD(df_agg, 'Elasticidad Suma')


        # Exportar a un archivo Excel
    archivo_salida = 'P103_mostrar_elasticidad_precio_demandaXX.xlsx'
    df_agg.to_excel(archivo_salida, index=False)  # Exporta el DataFrame a un archivo Excel


    # Crear el gráfico interactivo con Plotly (Elasticidad Precio-Demanda para Media y Suma de Total Volume)
    fig = go.Figure()

    # Desplazamiento para las líneas (ajustado por separado, ya que no es necesario como en las barras)
    offset = 0.2  # Ajuste de desplazamiento si es necesario, aunque para líneas no es crítico

    if dos_escalas:
        # Usamos dos escalas (izquierda para Elasticidad Media y derecha para Elasticidad Suma)
        fig.add_trace(go.Scatter(
            x=df_agg['Cal_AAAAMM_str'],
            y=df_agg['Elasticidad Media'],
            name='Elasticidad Media',
            mode='lines+markers',  # Modo línea con marcadores
            yaxis='y1',  # Asignamos al eje Y izquierdo
            line=dict(width=2),  # Ancho de la línea
        ))

        fig.add_trace(go.Scatter(
            x=df_agg['Cal_AAAAMM_str'],
            y=df_agg['Elasticidad Suma'],
            name='Elasticidad Suma',
            mode='lines+markers',  # Modo línea con marcadores
            yaxis='y2',  # Asignamos al eje Y derecho
            line=dict(width=2),  # Ancho de la línea
        ))

        # Ajustar el diseño para que las líneas aparezcan con dos escalas
        fig.update_layout(
            title=f'Elasticidad Precio-Demanda Agrupada ({agrupacion})',
            xaxis=dict(
                tickangle=45,
                title='Fecha'
            ),
            yaxis=dict(
                title='Elasticidad Media'  # Título para el eje izquierdo
            ),
            yaxis2=dict(
                title='Elasticidad Suma',  # Título para el eje derecho
                overlaying='y',  # Lo superpone al eje Y izquierdo
                side='right',  # Coloca la escala a la derecha
            ),
            height=600,  # Ajustar el tamaño del gráfico
        )

    else:
        # Usamos una sola escala para ambas líneas
        fig.add_trace(go.Scatter(
            x=df_agg['Cal_AAAAMM_str'],
            y=df_agg['Elasticidad Media'],
            name='Elasticidad Media',
            mode='lines+markers',  # Modo línea con marcadores
            yaxis='y1',  # Asignamos al eje Y izquierdo
            line=dict(width=2),  # Ancho de la línea
        ))

        fig.add_trace(go.Scatter(
            x=df_agg['Cal_AAAAMM_str'],
            y=df_agg['Elasticidad Suma'],
            name='Elasticidad Suma',
            mode='lines+markers',  # Modo línea con marcadores
            yaxis='y1',  # Asignamos al mismo eje Y izquierdo
            line=dict(width=2),  # Ancho de la línea
        ))

        # Ajustar el diseño para que las líneas aparezcan con una sola escala
        fig.update_layout(
            title=f'Elasticidad Precio-Demanda Agrupada ({agrupacion})',
            xaxis=dict(
                tickangle=45,
                title='Fecha'
            ),
            yaxis=dict(
                title='Elasticidad',  # Título para el eje izquierdo
            ),
            height=600,  # Ajustar el tamaño del gráfico
        )

    # Mostrar el gráfico
    fig.show()


# Función para calcular la elasticidad (puedes modificar la lógica según tus necesidades)
def calcular_elasticidadS(volumen, precio):
    # Aquí calculamos la elasticidad precio-demanda
    # Este cálculo es aproximado, usando un cambio porcentual entre el volumen y el precio
    cambio_volumen = volumen.pct_change()  # Cambio porcentual en volumen
    cambio_precio = precio.pct_change()  # Cambio porcentual en precio
    
    elasticidad = (cambio_volumen / cambio_precio) 
    


    return elasticidad

Data_ElasticidadMensuales =pd.DataFrame()

def Data01_MediaMensual_ElasticidadesSemanales(Datos):
    global Data_ElasticidadMensuales
        # Agrupar el DataFrame por 'Cal_AAAAMM_str' y calcular la media del campo 'Elasticidad'

    D = Datos.copy()
    # Reemplazar valores inf y NaN por 0 en el campo 'Elasticidad'
    Datos['Elasticidad'] = Datos['Elasticidad'].replace([float('inf'), -float('inf')], float('nan')).fillna(0)

    df_agg_elasticidad = D.groupby('Cal_AAAAMM_str').agg({'Elasticidad': 'mean'}).reset_index()

    # Renombrar la columna para mayor claridad
    df_agg_elasticidad.rename(columns={'Elasticidad': 'E Semanal Media Mensual'}, inplace=True)
    Data_ElasticidadMensuales = df_agg_elasticidad

    # Reemplazar valores inf y -inf por NaN en el campo 'Elasticidad'
    Datos['Elasticidad'] = Datos['Elasticidad'].replace([float('inf'), -float('inf')], float('nan'))
    
    # Eliminar las filas con valores NaN en el campo 'Elasticidad'
    Datos = Datos.dropna(subset=['Elasticidad'])


    df_agg_elasticidad = Datos.groupby('Cal_AAAAMM_str').agg({'Elasticidad': 'mean'}).reset_index()

    # Renombrar la columna para mayor claridad
    df_agg_elasticidad.rename(columns={'Elasticidad': 'E Semanal Media Mensual DROP'}, inplace=True)
    Data01_MediaMensual_ADD(df_agg_elasticidad,'E Semanal Media Mensual DROP')

def Data01_MediaMensual_ADD(Datos, CampoAdd):
    global Data_ElasticidadMensuales
    # Combinar ambos DataFrames y actualizar Data_ElasticidadMensuales
    Data_ElasticidadMensuales = Data_ElasticidadMensuales.merge(
    Datos[['Cal_AAAAMM_str', CampoAdd]],  # Seleccionar las columnas necesarias de Datos
    on='Cal_AAAAMM_str',  # Columna de relación
    how='left'  # Tipo de join (left join para mantener todas las filas de Data_ElasticidadMensuales)
    )

def graficar_Data_ElasticidadMensuales():
    """
    Genera un gráfico de líneas para 'Elasticidad Media', 'Elasticidad Suma' y 'E Semanal Media Mensual'
    con el eje X como 'Cal_AAAAMM_str' del DataFrame 'Data_ElasticidadMensuales'.
    """
    # Crear la figura
    fig = go.Figure()

    # Agregar las líneas para cada columna
    fig.add_trace(go.Scatter(
        x=Data_ElasticidadMensuales['Cal_AAAAMM_str'],
        y=Data_ElasticidadMensuales['Elasticidad Media'],
        name='Elasticidad Media',
        mode='lines+markers',  # Línea con marcadores
        line=dict(width=2)  # Ancho de la línea
    ))

    fig.add_trace(go.Scatter(
        x=Data_ElasticidadMensuales['Cal_AAAAMM_str'],
        y=Data_ElasticidadMensuales['Elasticidad Suma'],
        name='Elasticidad Suma',
        mode='lines+markers',  # Línea con marcadores
        line=dict(width=2)  # Ancho de la línea
    ))

    fig.add_trace(go.Scatter(
        x=Data_ElasticidadMensuales['Cal_AAAAMM_str'],
        y=Data_ElasticidadMensuales['E Semanal Media Mensual'],
        name='E Semanal Media Mensual',
        mode='lines+markers',  # Línea con marcadores
        line=dict(width=8)  # Ancho de la línea
    ))

    fig.add_trace(go.Scatter(
        x=Data_ElasticidadMensuales['Cal_AAAAMM_str'],
        y=Data_ElasticidadMensuales['E Semanal Media Mensual DROP'],
        name='E Semanal Media Mensual DROP',
        mode='lines+markers',  # Línea con marcadores
        line=dict(width=2)  # Ancho de la línea
    ))


    # Configuración del diseño del gráfico
    fig.update_layout(
        title='Elasticidades por Mes',
        xaxis=dict(
            title='Fecha (Cal_AAAAMM_str)',
            tickangle=45  # Angulo para las etiquetas del eje X
        ),
        yaxis=dict(
            title='Elasticidades',
            range=[-15, 15]  # Límites del eje Y            
        ),
        height=600,  # Altura del gráfico
    )

    # Mostrar el gráfico
    fig.show()

def graficar_Data_ElasticidadMensuales_DispersionMM():
    """
    Genera un gráfico de líneas para 'Elasticidad Media', 'Elasticidad Suma' y 'E Semanal Media Mensual'
    con el eje X como 'Cal_AAAAMM_str' del DataFrame 'Data_ElasticidadMensuales'.
    """
    # Crear la figura
    fig = go.Figure()

    # Convertir 'Cal_AAAAMM_str' a valores numéricos para realizar la regresión
    x = np.arange(len(Data_ElasticidadMensuales['Cal_AAAAMM_str']))  # Índices numéricos de las fechas
    y = Data_ElasticidadMensuales['E Semanal Media Mensual DROP']

    # Calcular la regresión polinómica (grado 2)
    coef = np.polyfit(x, y, 2)
    poly = np.poly1d(coef)
    y_poly = poly(x)

        # Calcular R^2
    r2 = r2_score(y, y_poly)

   # Agregar la línea de regresión polinómica al gráfico
    fig.add_trace(go.Scatter(
        x=Data_ElasticidadMensuales['Cal_AAAAMM_str'],
        y=y_poly,
        name='Regresión Polinómica (grado 2)',
        mode='lines',  # Solo líneas
        line=dict(width=2, color='red')  # Color y ancho de la línea
    ))

    fig.add_trace(go.Scatter(
        x=Data_ElasticidadMensuales['Cal_AAAAMM_str'],
        y=Data_ElasticidadMensuales['E Semanal Media Mensual DROP'],
        name='E Semanal Media Mensual DROP',
        #mode='lines+markers',  # Línea con marcadores
        mode='markers',  # Línea con marcadores
        #line=dict(width=2)  # Ancho de la línea
        marker=dict(size=4, color='blue')  # Tamaño de los puntos
    ))


    # Configuración del diseño del gráfico
    fig.update_layout(
        title='Elasticidades por Mes',
        xaxis=dict(
            title='Fecha (Cal_AAAAMM_str)',
            tickangle=45  # Angulo para las etiquetas del eje X
        ),
        yaxis=dict(
            title='Elasticidades',
            range=[-15, 15]  # Límites del eje Y            
        ),
        height=600,  # Altura del gráfico
    )

    # Mostrar el gráfico
    fig.show()   

    formula = f"\\(y = {coef[0]:.4f}x^2 + {coef[1]:.4f}x + {coef[2]:.4f}\\)"
    markdown_text = f"""
    ## Coeficientes de la Regresión Polinómica (Grado 2)
    - **Coeficiente cuadrático (a):** {coef[0]:.4f}
    - **Coeficiente lineal (b):** {coef[1]:.4f}
    - **Intersección (c):** {coef[2]:.4f}

    ### Fórmula de la Regresión:
    {formula}

    ### Coeficiente de Determinación (R²):
    - **R²:** {r2:.4f}

    """
    display(Markdown(markdown_text)) 


def graficar_Data_ElasticidadMensuales_Dispersion(pAgno ='2015'):
    """
    Genera un gráfico de líneas para 'Elasticidad Media', 'Elasticidad Suma' y 'E Semanal Media Mensual'
    con el eje X como 'Cal_AAAAMM_str' del DataFrame 'Data_ElasticidadMensuales'.
    """
    # Crear la figura
    fig = go.Figure()

    datos = DatosP103.copy()
    datos = datos[datos['CalYear'] == pAgno]

        # Reemplazar valores inf y NaN por 0 en el campo 'Elasticidad'
    datos['Elasticidad'] = datos['Elasticidad'].replace([float('inf'), -float('inf')], float('nan')).fillna(0)

    datos['Cal_AAAADDD_str'] = datos['Cal_AAAADDD'].astype(str)

    # Convertir 'Cal_AAAAMM_str' a valores numéricos para realizar la regresión
    x = np.arange(len(datos['Cal_AAAADDD'].unique()))  # Índices numéricos de las fechas
    y = datos['Elasticidad']

    # Calcular la regresión polinómica (grado 2)
    coef = np.polyfit(x, y, 2)
    poly = np.poly1d(coef)
    y_poly = poly(x)

        # Calcular R^2
    r2 = r2_score(y, y_poly)

   # Agregar la línea de regresión polinómica al gráfico
    fig.add_trace(go.Scatter(
        x=datos['Cal_AAAADDD_str'],
        y=y_poly,
        name='Regresión Polinómica (grado 2)',
        mode='lines',  # Solo líneas
        line=dict(width=2, color='red')  # Color y ancho de la línea
    ))

    fig.add_trace(go.Scatter(
        x=datos['Cal_AAAADDD_str'],
        y=datos['Elasticidad'],
        name='Elasticidad Semanal',
        #mode='lines+markers',  # Línea con marcadores
        mode='markers',  # Línea con marcadores
        #line=dict(width=2)  # Ancho de la línea
        marker=dict(size=5, color='blue')  # Tamaño de los puntos
    ))


    # Configuración del diseño del gráfico
    fig.update_layout(
        title='Elasticidades por granulda',
        xaxis=dict(
            title='Fecha (Cal_AAAADDD_str)',
            tickangle=45  # Angulo para las etiquetas del eje X
        ),
        yaxis=dict(
            title='Elasticidades',
            range=[-15, 15]  # Límites del eje Y            
        ),
        height=600,  # Altura del gráfico
    )

    # Mostrar el gráfico
    fig.show()   

    formula = f"\\(y = {coef[0]:.8f}x^2 + {coef[1]:.8f}x + {coef[2]:.8f}\\)"
    markdown_text = f"""
    ## Coeficientes de la Regresión Polinómica (Grado 2)
    - **Coeficiente cuadrático (a):** {coef[0]:.8f}
    - **Coeficiente lineal (b):** {coef[1]:.8f}
    - **Intersección (c):** {coef[2]:.8f}

    ### Fórmula de la Regresión:
    {formula}

    ### Coeficiente de Determinación (R²):
    - **R²:** {r2:.4f}

    """
    display(Markdown(markdown_text)) 

def graficar_Data_ElasticidadMensuales_DispersionORD01():
    """
    Genera un gráfico de líneas para 'Elasticidad Media', 'Elasticidad Suma' y 'E Semanal Media Mensual'
    con el eje X como 'Cal_AAAAMM_str' del DataFrame 'Data_ElasticidadMensuales'.
    """
    # Crear la figura
    fig = go.Figure()

    datos = DatosP103.copy()
    # Ordenar el DataFrame por la columna 'Elasticidad' de mayor a menor
        # Reemplazar valores inf y NaN por 0 en el campo 'Elasticidad'
    if 'reemplazar' =='':
        datos['Elasticidad'] = datos['Elasticidad'].replace([float('inf'), -float('inf')], float('nan')).fillna(0)
    else:
        # Reemplazar valores inf y -inf por NaN
        datos['Elasticidad'] = datos['Elasticidad'].replace([float('inf'), -float('inf')], float('nan'))

        # Eliminar filas con valores NaN en la columna 'Elasticidad'
        datos = datos.dropna(subset=['Elasticidad'])

    datos.sort_values(by='CambioVolumen', ascending=False, inplace=True)
    #datos = datos[datos['Elasticidad'] >= 0]
    

    datos['Cal_AAAADDD_str'] = datos['Cal_AAAADDD'].astype(str)

    # Convertir 'Cal_AAAAMM_str' a valores numéricos para realizar la regresión
    x = np.arange(len(datos['Cal_AAAADDD'].unique()))  # Índices numéricos de las fechas
    y = datos['Elasticidad']

        # Exportar a un archivo Excel
    archivo_salida = 'P103_graficar_Data_ElasticidadMensuales_DispersionORD_01.xlsx'
    datos.to_excel(archivo_salida, index=False)  # Exporta el DataFrame a un archivo Excel


    # Calcular la regresión polinómica (grado 2)
    coef = np.polyfit(x, y, 2)
    poly = np.poly1d(coef)
    y_poly = poly(x)

        # Calcular R^2
    r2 = r2_score(y, y_poly)

   # Agregar la línea de regresión polinómica al gráfico
    fig.add_trace(go.Scatter(
        #x=datos['Cal_AAAADDD_str'],
        #x=x,
        x=datos['CambioVolumen'],
        y=y_poly,
        name='Regresión Polinómica (grado 2)',
        mode='lines',  # Solo líneas
        line=dict(width=2, color='red')  # Color y ancho de la línea
    ))

    fig.add_trace(go.Scatter(
        #x=datos['Cal_AAAADDD_str'],
        x=datos['CambioVolumen'],
        y=datos['Elasticidad'],
        name='Elasticidad Semanal',
        #mode='lines+markers',  # Línea con marcadores
        mode='markers',  # Línea con marcadores
        #line=dict(width=2)  # Ancho de la línea
        marker=dict(size=5, color='blue')  # Tamaño de los puntos
    ))


    # Configuración del diseño del gráfico
    fig.update_layout(
        title='Elasticidades por granulda',
        xaxis=dict(
            title='Fecha (Cal_AAAADDD_str)',
            tickangle=45  # Angulo para las etiquetas del eje X
        ),
        yaxis=dict(
            title='Elasticidades',
            range=[-15, 15]  # Límites del eje Y            
        ),
        height=600,  # Altura del gráfico
    )

    # Mostrar el gráfico
    fig.show()   

    formula = f"\\(y = {coef[0]:.8f}x^2 + {coef[1]:.8f}x + {coef[2]:.8f}\\)"
    markdown_text = f"""
    ## Coeficientes de la Regresión Polinómica (Grado 2)
    - **Coeficiente cuadrático (a):** {coef[0]:.8f}
    - **Coeficiente lineal (b):** {coef[1]:.8f}
    - **Intersección (c):** {coef[2]:.8f}

    ### Fórmula de la Regresión:
    {formula}

    ### Coeficiente de Determinación (R²):
    - **R²:** {r2:.4f}

    """
    display(Markdown(markdown_text)) 


def graficar_Data_ElasticidadMensuales_DispersionORD02(pTipo = '%', pAgno =''):
    """
    Genera un gráfico de líneas para 'Elasticidad Media', 'Elasticidad Suma' y 'E Semanal Media Mensual'
    con el eje X como 'Cal_AAAAMM_str' del DataFrame 'Data_ElasticidadMensuales'.
    """
    # Crear la figura
    fig = go.Figure()

    datos = DatosP103.copy()

    if pAgno !='':
        datos = datos[datos['CalYear'] == pAgno]

    # Ordenar el DataFrame por la columna 'Elasticidad' de mayor a menor
        # Reemplazar valores inf y NaN por 0 en el campo 'Elasticidad'
    if 'reemplazar' =='':
        datos['Elasticidad'] = datos['Elasticidad'].replace([float('inf'), -float('inf')], float('nan')).fillna(0)
    else:
        # Reemplazar valores inf y -inf por NaN
        datos['Elasticidad'] = datos['Elasticidad'].replace([float('inf'), -float('inf')], float('nan'))

        # Eliminar filas con valores NaN en la columna 'Elasticidad'
        datos = datos.dropna(subset=['Elasticidad'])

    datos.sort_values(by='CambioVolumen', ascending=False, inplace=True)
    #datos = datos[datos['Elasticidad'] >= 0]
    

    datos['Cal_AAAADDD_str'] = datos['Cal_AAAADDD'].astype(str)

    # Convertir 'Cal_AAAAMM_str' a valores numéricos para realizar la regresión
    #x = np.arange(len(datos['Cal_AAAADDD'].unique()))  # Índices numéricos de las fechas
    #x = np.arange(len(datos['CambioVolumen'].unique()))  # Índices numéricos de las fechas
    # Valores únicos de 'CambioVolumen'
    vTipo ='Indice'

    if vTipo == '%':
        #datos['CambioVolumen'] = datos['CambioVolumen'] * 100
        x = datos['CambioVolumen']
    if vTipo == 'Indice':
        x = np.arange(len(datos['CambioVolumen']))  # Índices numéricos de las fechas

    datos['CambioVolumen'] = datos['CambioVolumen'] * 100
    x = datos['CambioVolumen']
    y = datos['Elasticidad']

        # Exportar a un archivo Excel
    archivo_salida = 'P103_graficar_Data_ElasticidadMensuales_DispersionORD_02.xlsx'
    datos.to_excel(archivo_salida, index=False)  # Exporta el DataFrame a un archivo Excel


    # Calcular la regresión polinómica (grado 2)
    coef = np.polyfit(x, y, 2)
    poly = np.poly1d(coef)
    y_poly = poly(x)

        # Calcular R^2
    r2 = r2_score(y, y_poly)

   # Agregar la línea de regresión polinómica al gráfico
    fig.add_trace(go.Scatter(
        #x=datos['Cal_AAAADDD_str'],
        x=x,
        y=y_poly,
        name='Regresión Polinómica (grado 2)',
        mode='lines',  # Solo líneas
        line=dict(width=2, color='red')  # Color y ancho de la línea
    ))

    fig.add_trace(go.Scatter(
        #x=datos['Cal_AAAADDD_str'],
        x=x,
        y=datos['Elasticidad'],
        name='Elasticidad Semanal',
        #mode='lines+markers',  # Línea con marcadores
        mode='markers',  # Línea con marcadores
        #line=dict(width=2)  # Ancho de la línea
        marker=dict(size=5, color='blue')  # Tamaño de los puntos
    ))


    # Configuración del diseño del gráfico
    fig.update_layout(
        title='Elasticidades por granulda',
        xaxis=dict(
            title=f'Ordenado por Cambio Volumen y Tipo={vTipo}',
            tickangle=45  # Angulo para las etiquetas del eje X
        ),
        yaxis=dict(
            title='Elasticidades',
            range=[-15, 15]  # Límites del eje Y            
        ),
        height=600,  # Altura del gráfico
    )

    # Mostrar el gráfico
    fig.show()   

    formula = f"\\(y = {coef[0]:.8f}x^2 + {coef[1]:.8f}x + {coef[2]:.8f}\\)"
    markdown_text = f"""
    ## Coeficientes de la Regresión Polinómica (Grado 2)
    - **Coeficiente cuadrático (a):** {coef[0]:.8f}
    - **Coeficiente lineal (b):** {coef[1]:.8f}
    - **Intersección (c):** {coef[2]:.8f}

    ### Fórmula de la Regresión:
    {formula}

    ### Coeficiente de Determinación (R²):
    - **R²:** {r2:.4f}

    """
    display(Markdown(markdown_text)) 

def graficar_Data_ElasticidadMensuales_DispersionDN01(pAgno='2015'):
    """
    Genera un gráfico de dispersión de 'Elasticidad' vs. 'CambioVolumen' 
    y ajusta una distribución normal.
    """
    # Crear la figura
    fig = go.Figure()

    datos = DatosP103.copy()

    # Reemplazar valores inf y NaN por NaN, y eliminar valores faltantes
    datos['Elasticidad'] = datos['Elasticidad'].replace([float('inf'), -float('inf')], float('nan'))
    datos = datos.dropna(subset=['Elasticidad'])

    # Ordenar los datos
    datos.sort_values(by='CambioVolumen', ascending=False, inplace=True)

    # Calcular parámetros de la distribución normal
    mu, sigma = norm.fit(datos['Elasticidad'])  # Media y desviación estándar
    print(f"Media (mu): {mu:.4f}, Desviación estándar (sigma): {sigma:.4f}")

    # Crear valores x para la curva normal
    x = np.linspace(datos['Elasticidad'].min(), datos['Elasticidad'].max(), 500)
    y_norm = norm.pdf(x, mu, sigma)  # Densidad de probabilidad

    # Agregar la curva de la distribución normal ajustada al gráfico
    fig.add_trace(go.Scatter(
        x=x,
        y=y_norm,
        name='Distribución Normal Ajustada',
        mode='lines',
        line=dict(width=2, color='red')  # Línea roja para la curva
    ))

    # Agregar los datos originales al gráfico
    fig.add_trace(go.Scatter(
        x=datos['Elasticidad'],
        y=np.zeros_like(datos['Elasticidad']),  # Agregar puntos para visualización
        name='Elasticidad Semanal',
        mode='markers',
        marker=dict(size=5, color='blue')  # Puntos azules
    ))

    # Configuración del diseño del gráfico
    fig.update_layout(
        title='Elasticidades Ajustadas a una Distribución Normal',
        xaxis=dict(
            title='Elasticidad',
            tickangle=45  # Ángulo para las etiquetas del eje X
        ),
        yaxis=dict(
            title='Densidad de Probabilidad',
            range=[0, max(y_norm) * 1.1]  # Ajustar el rango Y al máximo de la curva
        ),
        height=600,  # Altura del gráfico
    )

    # Mostrar el gráfico
    fig.show()

    # Mostrar parámetros de la distribución en formato Markdown
    markdown_text = f"""
    ## Parámetros de la Distribución Normal Ajustada
    - **Media (μ):** {mu:.4f}
    - **Desviación Estándar (σ):** {sigma:.4f}

    ### Fórmula de la Densidad de Probabilidad:
    \\[
    f(x) = \\frac{{1}}{{\\sqrt{{2\\pi \\sigma^2}}}} e^{{-\\frac{{(x - \\mu)^2}}{{2\\sigma^2}}}}
    \\]
    """
    display(Markdown(markdown_text))


    # Función logística
def logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

def graficar_Data_ElasticidadMensuales_DispersionLOG01(pAgno='2015'):
    """
    Genera un gráfico de dispersión de 'Elasticidad' vs. 'CambioVolumen'
    y ajusta una curva logística.
    """
    # Crear la figura
    fig = go.Figure()

    datos = DatosP103.copy()

    # Reemplazar valores inf y NaN por NaN, y eliminar valores faltantes
    datos['Elasticidad'] = datos['Elasticidad'].replace([float('inf'), -float('inf')], float('nan'))
    datos = datos.dropna(subset=['Elasticidad'])

    # Ordenar los datos
    datos.sort_values(by='CambioVolumen', ascending=False, inplace=True)

    # Datos de entrada para el ajuste
    x_data = datos['CambioVolumen'].values
    y_data = datos['Elasticidad'].values

    # Estimación inicial de parámetros [L, k, x0]
    initial_guess = [y_data.max(), 1, x_data.mean()]

    # Ajuste de la curva logística
    params, covariance = curve_fit(logistic, x_data, y_data, p0=initial_guess)
    L, k, x0 = params  # Parámetros ajustados

    # Generar datos para la curva ajustada
    x_curve = np.linspace(x_data.min(), x_data.max(), 500)
    y_curve = logistic(x_curve, L, k, x0)

    # Agregar la curva logística ajustada al gráfico
    fig.add_trace(go.Scatter(
        x=x_curve,
        y=y_curve,
        name='Tendencia Logística',
        mode='lines',
        line=dict(width=2, color='red')  # Línea roja para la tendencia logística
    ))

    # Agregar los datos originales al gráfico
    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data,
        name='Elasticidad Semanal',
        mode='markers',
        marker=dict(size=5, color='blue')  # Puntos azules
    ))

    # Configuración del diseño del gráfico
    fig.update_layout(
        title='Elasticidades con Ajuste Logístico',
        xaxis=dict(
            title='CambioVolumen',
            tickangle=45  # Ángulo para las etiquetas del eje X
        ),
        yaxis=dict(
            title='Elasticidad',
            range=[min(y_data) - 1, max(y_data) + 1]  # Ajustar el rango Y para mejor visualización
        ),
        height=600,  # Altura del gráfico
    )

    # Mostrar el gráfico
    fig.show()

    # Mostrar parámetros del ajuste logístico en formato Markdown
    markdown_text = f"""
    ## Parámetros de la Tendencia Logística
    - **Valor máximo (L):** {L:.4f}
    - **Pendiente (k):** {k:.4f}
    - **Punto medio (x₀):** {x0:.4f}

    ### Fórmula de la Tendencia Logística:
    \\[
    f(x) = \\frac{{{L:.4f}}}{{1 + e^{{-{k:.4f}(x - {x0:.4f})}}}}
    \\]
    """
    display(Markdown(markdown_text))