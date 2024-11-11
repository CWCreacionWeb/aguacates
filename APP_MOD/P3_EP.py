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
#vEnunciados = Enunciados()
# --------------------- 3. Elasticidad del Precio ---------------------
# Función para calcular la elasticidad
def calcular_elasticidad(volumen, precio):

    # Calcular el porcentaje de cambio en volumen y precio
    cambio_volumen = volumen.pct_change()
    cambio_precio = precio.pct_change()
    # Calcular elasticidad precio-demanda
    elasticidad = (cambio_volumen / cambio_precio)#.fillna(0)



    return elasticidad, cambio_volumen, cambio_precio


def P3_1_Elasticidad_Precio_Demanda_Año(pListaRegiones =''):

    APP_Enunciados.getEnunciado('3.1')

    mDbg = f"""
- **parametros**:  
     - **pListaRegiones:** `{pListaRegiones}`

    """

    display(Markdown(mDbg))



    SubData= Datos
    if(pListaRegiones==''):
        SubData = SubData
    else:
        SubData = SubData[Datos['region'].isin(pListaRegiones)]  

    Datos_anual = SubData.groupby('CalYear').agg({'Total Volume': 'sum', 'AveragePrice': 'mean'}).reset_index()


    #Datos_anual['Elasticidad'] = calcular_elasticidad(Datos_anual['Total Volume'], Datos_anual['AveragePrice'])
     # Calcular la elasticidad y cambios en volumen y precio
    Datos_anual['Elasticidad'], Datos_anual['Cambio_Volumen'], Datos_anual['Cambio_Precio'] =  calcular_elasticidad(Datos_anual['Total Volume'], Datos_anual['AveragePrice'])


    print('Tabla Elasticidad periodo')
    print(Datos_anual[['Cambio_Volumen','Cambio_Precio','Elasticidad']])
    # Gráfico de elasticidad por año
    plt.figure(figsize=(12, 6))
    plt.plot(Datos_anual['CalYear'], Datos_anual['Elasticidad'], marker='o', color='b')
    plt.title('Elasticidad cambio_volumen / cambio_precio por Año')
    plt.xlabel('Año')
    plt.ylabel('Elasticidad')
    plt.grid(True)

    # Añadir el texto en cada punto del gráfico
    for i, row in Datos_anual.iterrows():
        year = row['CalYear']
        elasticidad = row['Elasticidad']
        cambio_volumen = row['Cambio_Volumen']
        cambio_precio = row['Cambio_Precio']
        texto_info=''
        if math.isnan(elasticidad)==False: 
            texto_info = f"Info: {cambio_volumen*100:.2f}% / {cambio_precio*100:.2f}% = {elasticidad:.2f}"
        plt.text(year, elasticidad, texto_info, ha='right', va='bottom', fontsize=9, color='purple')

  # Asegurarse de que los años en el eje x se muestren como enteros
    plt.xticks(Datos_anual['CalYear'], rotation=45)  # Los años se muestran sin decimales y alineados verticalmente

    plt.show()


def P3_2_Elasticidad_Regiones():
    """
2. **Comparación de Elasticidad en Diferentes Mercados:**
   - **Uso de Datos:** Utiliza las columnas `Total Volume` y `AveragePrice`.
   - **Esperado:** Calcula la elasticidad del precio de la demanda en diferentes regiones.
     - Agrupa los datos por `region` y calcula la elasticidad para cada región utilizando `pd.pct_change()`.
     - Presenta un gráfico de barras que muestre la elasticidad por región usando `plt.bar()`.    
    """
    mDbg =P3_2_Elasticidad_Regiones.__doc__

    display(Markdown(mDbg))

    print("Comparando Elasticidad en Diferentes Regiones...")
    # Agrupar datos por región y calcular la elasticidad para cada región
    SubData = Datos
    Datos_region = SubData.groupby('CalYear','region').agg({'Total Volume': 'sum', 'AveragePrice': 'mean'}).reset_index()

    Datos_region['Elasticidad'] = calcular_elasticidad(Datos_region['Total Volume'], Datos_region['AveragePrice'])
         # Calcular la elasticidad y cambios en volumen y precio
    #Datos_anual['Elasticidad'], Datos_anual['Cambio_Volumen'], Datos_anual['Cambio_Precio'] =  calcular_elasticidad(Datos_anual['Total Volume'], Datos_anual['AveragePrice'])

    # Gráfico de elasticidad por región
    plt.figure(figsize=(12, 6))
    plt.bar(Datos_region['region'], Datos_region['Elasticidad'], color='skyblue')
    plt.title('Elasticidad Precio-Demanda por Región')
    plt.xlabel('Región')
    plt.ylabel('Elasticidad')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()



def calcular_elasticidadB(volumen, precio):
    """
    Calcula la elasticidad precio-demanda como el cambio porcentual en el volumen
    dividido por el cambio porcentual en el precio.
    """
    # Evitar division por cero
    if volumen.empty or precio.empty:
        return pd.Series([0]*len(volumen))

    cambio_volumen = volumen.pct_change().fillna(0)  # Cambio porcentual en el volumen
    cambio_precio = precio.pct_change().fillna(0)    # Cambio porcentual en el precio

    # Calculamos la elasticidad, evitando divisiones por cero
    elasticidad = cambio_volumen / cambio_precio.replace(0, 1)  # Reemplaza 0 en cambio_precio por 1 para evitar division por cero
    return elasticidad

def P3_3_Elasticidad_BolsasB():
    APP_Enunciados.getEnunciado("3.3")
    APP_Enunciados.getExplicacion("3.3")

    print("Calculando Elasticidad media de los 3 años,  para Cada Tipo de Bolsa...")
    global Datos
    # Agrupar y sumar los volúmenes de cada tipo de bolsa por año y calcular el precio promedio
    Datos_bolsas = Datos.groupby('CalYear').agg({
        'AveragePrice': 'mean', 
        'Small Bags': 'sum', 
        'Large Bags': 'sum', 
        'XLarge Bags': 'sum'
    }).reset_index()

    print(Datos_bolsas)

    cambio_volumen = Datos_bolsas['Small Bags'].pct_change().fillna(0)  # Cambio porcentual en el volumen
    print(cambio_volumen)

    # Calcular la elasticidad para cada tipo de bolsa
    Datos_bolsas['Elasticidad_Small'] = calcular_elasticidadB(Datos_bolsas['Small Bags'], Datos_bolsas['AveragePrice'])
    Datos_bolsas['Elasticidad_Large'] = calcular_elasticidadB(Datos_bolsas['Large Bags'], Datos_bolsas['AveragePrice'])
    Datos_bolsas['Elasticidad_XLarge'] = calcular_elasticidadB(Datos_bolsas['XLarge Bags'], Datos_bolsas['AveragePrice'])

    
    print(Datos_bolsas['Small Bags'])
    print(Datos_bolsas['AveragePrice'])
    print(Datos_bolsas['Elasticidad_Small'])

    # Calcular la elasticidad promedio
    elasticidades_promedio = [
        Datos_bolsas['Elasticidad_Small'].mean(),
        Datos_bolsas['Elasticidad_Large'].mean(),
        Datos_bolsas['Elasticidad_XLarge'].mean()
    ]

    # Graficar las elasticidades de cada tipo de bolsa
    plt.figure(figsize=(12, 6))
    plt.bar(['Small Bags', 'Large Bags', 'XLarge Bags'], elasticidades_promedio, color=['blue', 'orange', 'green'])
    
    plt.title('Elasticidad Precio-Demanda por Tipo de Bolsa')
    plt.xlabel('Tipo de Bolsa')
    plt.ylabel('Elasticidad Promedio')
    plt.show()
    
def P3_4_Elasticidad_Tipo():
    APP_Enunciados.getEnunciado("3.4")
    APP_Enunciados.getExplicacion("3.4")


    # Agrupar datos por año y tipo de aguacate, y calcular volumen total y precio promedio
    Datos_tipo = Datos.groupby(['CalYear', 'type']).agg({
        'Total Volume': 'sum', 
        'AveragePrice': 'mean'
    }).reset_index()


    
    
    # Calcular cambios porcentuales
    Datos_tipo['Cambio_Volumen'] = Datos_tipo.groupby('type')['Total Volume'].pct_change()
    Datos_tipo['Cambio_Precio'] = Datos_tipo.groupby('type')['AveragePrice'].pct_change()
    
    # Calcular la elasticidad
    Datos_tipo['Elasticidad'] = Datos_tipo['Cambio_Volumen'] / Datos_tipo['Cambio_Precio']
    
    # Eliminar filas con valores NaN (primera fila de cada grupo)
    Datos_tipo = Datos_tipo.dropna(subset=['Elasticidad'])

    # Gráfico comparativo de elasticidad entre tipos de aguacates
    plt.figure(figsize=(10, 6))
    """
    for tipo in Datos_tipo['type'].unique():
        subset = Datos_tipo[Datos_tipo['type'] == tipo]
        plt.plot(subset['CalYear'].astype(str), subset['Elasticidad'], marker='o', label=f'{tipo}')
    """
        # Configuración del gráfico
    plt.figure(figsize=(12, 6))
    años = Datos_tipo['CalYear'].unique()
    ancho_barra = 0.35  # Ancho de las barras
    # Generar el gráfico de barras para cada tipo (orgánico y convencional)
    for i, tipo in enumerate(Datos_tipo['type'].unique()):
        subset = Datos_tipo[Datos_tipo['type'] == tipo]
        # Posiciones en el eje X con desplazamiento para evitar superposición
        posiciones_x = subset['CalYear'] + (i * ancho_barra) - ancho_barra / 2
        plt.bar(posiciones_x, subset['Elasticidad'], width=ancho_barra, label=f'{tipo}')


    plt.title('Elasticidad Comparativa: Orgánicos vs Convencionales')
    plt.xlabel('Año')
    plt.ylabel('Elasticidad')
    plt.legend(title="Tipo de Aguacate")
    plt.grid(True)
    plt.show()




# Punto 3.5 Análisis de la Elasticidad Precios-Ventas
def P3_5_Elasticidad_Precio_Ventas():
    APP_Enunciados.getEnunciado("3.5")
    APP_Enunciados.getExplicacion("3.5")
    global Datos
    MisDatos = Datos.copy()



    print("Analizando Elasticidad entre Precios y Ventas Totales...")
    # Calcular elasticidad entre precio promedio y volumen total
    elasticidad = calcular_elasticidadB(MisDatos['Total Volume'], MisDatos['AveragePrice'])
    MisDatos['Elasticidad_Precio_Ventas'] = elasticidad
    # Gráfico de dispersión de la relación entre precio y volumen
    plt.figure(figsize=(10, 6))
    plt.scatter(MisDatos['AveragePrice'], MisDatos['Total Volume']/1000,MisDatos['Elasticidad_Precio_Ventas'], alpha=0.5, color='purple')
    sns.regplot(data=Datos, x='AveragePrice', y='Total Volume', scatter=False, color='red')

    plt.title('Relación entre Precio y Volumen de Ventas')
    plt.xlabel('Precio Promedio')
    plt.ylabel('Volumen Total (miles)')
    plt.grid(True)
    plt.show()

def P3_Precios_Promedio_Mensuales():
    # 1. Comparación de Precios Promedio Mensuales
    # Agrupación por mes y cálculo del precio promedio
    monthly_avg_price = Datos.groupby(pd.Grouper(key='Fecha', freq='M'))['AveragePrice'].mean()

    # Visualización de la comparación de precios promedio mensuales
    plt.figure(figsize=(10, 6))
    plt.plot(monthly_avg_price.index, monthly_avg_price.values, marker='o', color='b', label='Precio Promedio Mensual')
    plt.xlabel('Fecha')
    plt.ylabel('Precio Promedio')
    plt.title('Comparación de Precios Promedio Mensuales')
    plt.legend()
    plt.grid()
    plt.show()




def STP_Visualizar():
    # Agrupar por región y mes para obtener el precio promedio mensual
    grouped_data = Datos.groupby([ 'Date'])['AveragePrice'].mean().reset_index()
    # Configuración de los gráficos
    plt.figure(figsize=(14, 10))
    # Graficar cada región
    plt.plot(grouped_data['Date'], grouped_data['AveragePrice'], label=grouped_data['Date'])

    # Configurar detalles del gráfico
    plt.title('Tendencias Estacionales de Precios de Aguacates por Región')
    plt.xlabel('Mes')
    plt.ylabel('Precio Promedio')
    plt.xticks(range(1, 13), ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
    plt.legend(title='Región')
    plt.grid()
    plt.show()




