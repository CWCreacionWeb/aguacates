
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

# --------------------- 3. Elasticidad del Precio ---------------------
# Función para calcular la elasticidad
def calcular_elasticidad(volumen, precio):
    # Calcular el porcentaje de cambio en volumen y precio
    cambio_volumen = volumen.pct_change()
    cambio_precio = precio.pct_change()
    # Calcular elasticidad precio-demanda
    elasticidad = (cambio_volumen / cambio_precio).fillna(0)
    return elasticidad

def P3_1_Elasticidad_Precio_Demanda_Año(pListaRegiones =''):
    print("Calculando Elasticidad Precio-Demanda por Año...")
    # Agrupar datos por año y calcular la elasticidad anual
    #SubDatos = datos['region'] =''
    if(pListaRegiones==''):
        SubData = Datos
    else:
        SubData = Datos[Datos['region'] == 'TotalUS']
    Datos_anual = SubData.groupby('CalYear').agg({'Total Volume': 'sum', 'AveragePrice': 'mean'}).reset_index()
    Datos_anual['Elasticidad'] = calcular_elasticidad(Datos_anual['Total Volume'], Datos_anual['AveragePrice'])
    # Gráfico de elasticidad por año
    plt.figure(figsize=(12, 6))
    plt.plot(Datos_anual['CalYear'], Datos_anual['Elasticidad'], marker='o', color='b')
    plt.title('Elasticidad Precio-Demanda por Año')
    plt.xlabel('Año')
    plt.ylabel('Elasticidad')
    plt.grid(True)
  # Asegurarse de que los años en el eje x se muestren como enteros
    plt.xticks(Datos_anual['CalYear'], rotation=45)  # Los años se muestran sin decimales y alineados verticalmente

    plt.show()


def P3_2_Elasticidad_Regiones():
    print("Comparando Elasticidad en Diferentes Regiones...")
    # Agrupar datos por región y calcular la elasticidad para cada región
    Datos_region = Datos.groupby('region').agg({'Total Volume': 'sum', 'AveragePrice': 'mean'}).reset_index()
    Datos_region['Elasticidad'] = calcular_elasticidad(Datos_region['Total Volume'], Datos_region['AveragePrice'])
    # Gráfico de elasticidad por región
    plt.figure(figsize=(12, 6))
    plt.bar(Datos_region['region'], Datos_region['Elasticidad'], color='skyblue')
    plt.title('Elasticidad Precio-Demanda por Región')
    plt.xlabel('Región')
    plt.ylabel('Elasticidad')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# Punto 3.3 Elasticidad a Nivel de Tipo de Bolsa
def P3_3_Elasticidad_Bolsas():
    print("Calculando Elasticidad para Cada Tipo de Bolsa...")
    # Sumar volúmenes de cada tipo de bolsa por año y calcular elasticidad
    Datos_bolsas = Datos.groupby('CalYear').agg({'Total Bags': 'sum', 'AveragePrice': 'mean'}).reset_index()
    Datos_bolsas['Elasticidad'] = calcular_elasticidad(Datos_bolsas['Total Bags'], Datos_bolsas['AveragePrice'])
    # Gráfico comparativo de elasticidad para cada tipo de bolsa
    plt.figure(figsize=(10, 6))
    plt.bar(Datos_bolsas['CalYear'].astype(str), Datos_bolsas['Elasticidad'], color='blue')
    plt.title('Elasticidad Precio-Demanda por Tipo de Bolsa')
    plt.xlabel('Año')
    plt.ylabel('Elasticidad')
    plt.show()

# Punto 3.4 Análisis de Elasticidad Comparativa entre Orgánicos y Convencionales
def P3_4_Elasticidad_Tipo():
    print("Comparando Elasticidad entre Aguacates Orgánicos y Convencionales...")
    # Calcular elasticidad para aguacates orgánicos y convencionales
    Datos_tipo = Datos.groupby(['CalYear', 'type']).agg({'Total Volume': 'sum', 'AveragePrice': 'mean'}).reset_index()
    Datos_tipo['Elasticidad'] = Datos_tipo.groupby('type').apply(lambda x: calcular_elasticidad(x['Total Volume'], x['AveragePrice'])).reset_index(level=0, drop=True)
    # Gráfico comparativo de elasticidad entre tipos de aguacates
    plt.figure(figsize=(10, 6))
    for tipo in Datos_tipo['type'].unique():
        subset = Datos_tipo[Datos_tipo['type'] == tipo]
        plt.plot(subset['CalYear'].astype(str), subset['Elasticidad'], marker='o', label=f'{tipo}')
    plt.title('Elasticidad Comparativa: Orgánicos vs Convencionales')
    plt.xlabel('Año')
    plt.ylabel('Elasticidad')
    plt.legend()
    plt.show()

# Punto 3.5 Análisis de la Elasticidad Precios-Ventas
def P3_5_Elasticidad_Precio_Ventas():
    print("Analizando Elasticidad entre Precios y Ventas Totales...")
    # Calcular elasticidad entre precio promedio y volumen total
    elasticidad = calcular_elasticidad(Datos['Total Volume'], Datos['AveragePrice'])
    Datos['Elasticidad_Precio_Ventas'] = elasticidad
    # Gráfico de dispersión de la relación entre precio y volumen
    plt.figure(figsize=(10, 6))
    plt.scatter(Datos['AveragePrice'], Datos['Total Volume'], alpha=0.5, color='purple')
    plt.title('Relación entre Precio y Volumen de Ventas')
    plt.xlabel('Precio Promedio')
    plt.ylabel('Volumen Total')
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




def prueba():
        # Agrupar los datos por fecha y calcular el precio promedio diario
    avg_price_daily = Datos.groupby('Date')['AveragePrice'].mean()
    avg_price_daily = Datos.groupby('Fecha')['AveragePrice'].mean()
    #print(avg_price_daily)
    mDbg =f'Agrupación dias:{avg_price_daily.__len__()}\n'
    #mDbg +=f'Fecha minima:{avg_price_daily.index.tolist}\n'
    mDbg +=f'Fecha minima:{avg_price_daily.iloc[0]}\n'
    mDbg +=f'xFecha maxima:{avg_price_daily.iloc[-1]}\n'
    mDbg +=f'180\n'
    print(mDbg)
    """
        valores_fecha = avg_price_daily.index.tolist()
        # Encontrar el elemento máximo y mínimo del índice
        elemento_maximo = max(valores_fecha)
        elemento_minimo = min(valores_fecha)
        e0 = avg_price_daily.index[0]
        #s1= e0.strftime("%d-%m-%Y")
        s0 = e0.strftime("%d-%m-%Y")
        mDbg +=f'Fecha maxima:{avg_price_daily.index.__len__()}\n'
        mDbg +=f'Fecha maxima:{avg_price_daily.index[0].strftime("%d-%m-%Y")}\n'
        mDbg +=f'Fecha maxima:{avg_price_daily.index[-1].strftime("%d-%m-%Y")}\n'
    """ 
    # Realizar la descomposición de la serie temporal
    decomposition = seasonal_decompose(avg_price_daily, model='additive', period=180)  # Ajuste 'period' si es necesario
    print('180')
            

    # Graficar los componentes
    plt.figure(figsize=(14, 28))
    plt.subplot(411)

    plt.title('TITULO')
    plt.xlabel('XLABEL')
    plt.ylabel('YLABEL')
    plt.grid(axis='y')
    plt.xticks(rotation=45)
    plt.legend(loc='upper left')
    #plt.style.use('dark_background')

    #plt.plot(color='red')
    plt.show()

def ignore_nan(arr):
    return max(filter(lambda x: not math.isnan(x), arr))
def P11_DST_TEST(pPeriod):
    decomposition =seasonal_decompose(avg_price_daily, model='additive', period=pPeriod)
    plt.rcParams["figure.figsize"] = (10,10)
    fig = decomposition.plot()
    plt.show()
class P11_DST:
    #PrepararDatos([30,52,80])
    global avg_price_daily
    mPeriodos=[]
    #Level
    #Trend
    #Season
    #Noise
    mTipo='Level'
    mTipoDesc='Level'
    def __init__(self):
        self.mPeriodos = [52]
        self.mTipo = 'Trend'
    def Periodos(self, pPeriodos):
        self.mPeriodos = pPeriodos
    def MostrarGrafico(self):
        max = 0
        plt.figure(figsize=(14, 10))
        periodoMax=0
        #for periodo in range(80,1,-7):
        for periodo in self.mPeriodos:
            print(f'period={periodo}')
            decomposition = seasonal_decompose(avg_price_daily, model='additive', period=periodo)  # Ajuste 'period' si es necesario
            if periodo < 30:
                vColor = 'black'
            elif periodo < 52:
                vColor = 'green'
            elif periodo == 52:
                vColor = 'red'
            elif periodo > 52:
                vColor = 'blue'
            if self.mTipo =='trend':
                plt.plot(decomposition.trend,color=vColor, label=f'Tendencia {periodo}')
                mTipoDesc=''
            elif self.mTipo=='observed':
                plt.plot(decomposition.observed,color=vColor, label=f'Tendencia {periodo}')
            elif self.mTipo=='seasonal':
                plt.plot(decomposition.seasonal,color=vColor, label=f'Tendencia {periodo}')
            elif self.mTipo=='resid':
                plt.plot(decomposition.resid,color=vColor, label=f'Tendencia {periodo}')
        plt.title(f'Componente de {self.mTipo}')
        plt.xlabel('Fecha')
        plt.ylabel('Precio')
        plt.xticks(rotation=45)
        plt.legend(loc='upper left')
        plt.grid(axis='y')  # Cuadrícula horizontal
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))    
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))    
        plt.tight_layout()
        plt.show()
        
    
# ejemplo P1_Proceso( [1,30,52,60])
def P1_Proceso( pLista):
    max = 0
    
    plt.figure(figsize=(14, 28))
    periodoMax=0
    #for periodo in range(80,1,-7):
    for periodo in pLista:
        print(f'period={periodo}')
        decomposition = seasonal_decompose(avg_price_daily, model='additive', period=periodo)  # Ajuste 'period' si es necesario
        if periodo < 30:
            vColor = 'black'
        elif periodo < 52:
            vColor = 'green'
        elif periodo == 52:
            vColor = 'red'
        elif periodo > 52:
            vColor = 'blue'

        plt.plot(decomposition.trend,color=vColor, label=f'Tendencia {periodo}')
    plt.title(f'Componente de Tendencia')
    plt.xlabel('Fecha')
    plt.ylabel('Precio')
    plt.xticks(rotation=45)
    plt.legend(loc='upper left')
    plt.grid(axis='y')  # Cuadrícula horizontal
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))    
    plt.tight_layout()
    plt.show()

def P1_ProcesoRuido():
    max = 0
    
    plt.figure(figsize=(14, 28))
    periodoMax=0
    #for periodo in range(80,1,-7):
    for periodo in[1,30,52,60]:
        print(f'period={periodo}')
        decomposition = seasonal_decompose(avg_price_daily, model='additive', period=periodo)  # Ajuste 'period' si es necesario
        vArr = decomposition.resid.values
        maxAux = ignore_nan(vArr)

        if periodo == 1:
            vColor = 'black'
        elif periodo == 30:
            vColor = 'green'
        elif periodo == 52:
            vColor = 'red'
        elif periodo == 60:
            vColor = 'blue'

        plt.plot(decomposition.resid,color=vColor, label=f'Tendencia {periodo}')
    plt.title(f'Componente de Tendencia')
    plt.xlabel('Fecha')
    plt.ylabel('Precio')
    plt.xticks(rotation=45)
    plt.legend(loc='upper left')
    plt.grid(axis='y')  # Cuadrícula horizontal
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))    
    plt.tight_layout()
    plt.show()



def P1_Series_Temporales_Precios():
    # Agrupar los datos por fecha y calcular el precio promedio diario
    avg_price_daily = Datos.groupby('Date')['AveragePrice'].mean()
    avg_price_daily = Datos.groupby('Fecha')['AveragePrice'].mean()
    #print(avg_price_daily)
    mDbg =f'Agrupación dias:{avg_price_daily.__len__()}\n'
    #mDbg +=f'Fecha minima:{avg_price_daily.index.tolist}\n'
    mDbg +=f'Fecha minima:{avg_price_daily.iloc[0]}\n'
    mDbg +=f'Fecha maxima:{avg_price_daily.iloc[-1]}\n'
    print(mDbg)
    """
        valores_fecha = avg_price_daily.index.tolist()
        # Encontrar el elemento máximo y mínimo del índice
        elemento_maximo = max(valores_fecha)
        elemento_minimo = min(valores_fecha)
        e0 = avg_price_daily.index[0]
        #s1= e0.strftime("%d-%m-%Y")
        s0 = e0.strftime("%d-%m-%Y")
        mDbg +=f'Fecha maxima:{avg_price_daily.index.__len__()}\n'
        mDbg +=f'Fecha maxima:{avg_price_daily.index[0].strftime("%d-%m-%Y")}\n'
        mDbg +=f'Fecha maxima:{avg_price_daily.index[-1].strftime("%d-%m-%Y")}\n'
    """ 
    # Realizar la descomposición de la serie temporal
    decomposition26 = seasonal_decompose(avg_price_daily, model='additive', period=26)  # Ajuste 'period' si es necesario
    decomposition52 = seasonal_decompose(avg_price_daily, model='additive', period=52)  # Ajuste 'period' si es necesario
    decomposition = seasonal_decompose(avg_price_daily, model='additive', period=4)  # Ajuste 'period' si es necesario
    print('period=360/7')
    plt.figure(figsize=(14, 28))
    plt.subplot(221)
    plt.plot(decomposition52.observed,color='red', label='observed')
    plt.title('Componente de observed 52')
    plt.xlabel('Fecha')
    plt.ylabel('Precio')
    plt.xticks(rotation=45)
    plt.legend(loc='upper left')
    plt.grid(axis='y')  # Cuadrícula horizontal
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))    
    plt.subplot(222)
    plt.plot(decomposition26.observed,color='red', label='observed')
    plt.title('Componente de observed 26')
    plt.xlabel('Fecha')
    plt.ylabel('Precio')
    plt.xticks(rotation=45)
    plt.legend(loc='upper left')
    plt.grid(axis='y')  # Cuadrícula horizontal
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))    
    plt.tight_layout()
    plt.show()



    plt.figure(figsize=(14, 28))
    plt.subplot(221)
    plt.plot(decomposition52.trend,color='red', label='Tendencia')
    plt.title('Componente de Tendencia 52')
    plt.xlabel('Fecha')
    plt.ylabel('Precio')
    plt.xticks(rotation=45)
    plt.legend(loc='upper left')
    plt.grid(axis='y')  # Cuadrícula horizontal
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))    
    plt.subplot(222)
    plt.plot(decomposition26.trend,color='red', label='Tendencia')
    plt.title('Componente de Tendencia 26')
    plt.xlabel('Fecha')
    plt.ylabel('Precio')
    plt.xticks(rotation=45)
    plt.legend(loc='upper left')
    plt.grid(axis='y')  # Cuadrícula horizontal
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))    
            

    plt.figure(figsize=(14, 28))
    plt.subplot(221)
    plt.plot(decomposition52.seasonal,color='red', label='Estacionalidad')
    plt.title('Componente de Estacionalidad 52')
    plt.xlabel('Fecha')
    plt.ylabel('Precio')
    plt.xticks(rotation=45)
    plt.legend(loc='upper left')
    plt.grid(axis='y')  # Cuadrícula horizontal
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))    
    plt.subplot(222)
    plt.plot(decomposition26.seasonal,color='red', label='Estacionalidad')
    plt.title('Componente de Estacionalidad 26')
    plt.xlabel('Fecha')
    plt.ylabel('Precio')
    plt.xticks(rotation=45)
    plt.legend(loc='upper left')
    plt.grid(axis='y')  # Cuadrícula horizontal
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))    

    # Graficar los componentes
    plt.figure(figsize=(14, 28))
    #plt.subplot(411)
    #plt.style.use('dark_background')

    #plt.plot(color='red')
    #plt.show()

    plt.title('TITULO')
    plt.xlabel('XLABEL')
    plt.ylabel('YLABEL')
    plt.grid(axis='y')
    plt.xticks(rotation=45)
    plt.legend(loc='upper left')

    plt.plot(decomposition.observed, label='Original')
        
    #plt.title('TITULO')
    #plt.xlabel('XLABEL')
    #plt.ylabel('YLABEL')
    #plt.grid(axis='y')
    #plt.xticks(rotation=45)
    #plt.legend(loc='upper left')
    plt.subplot(412)
    plt.plot(decomposition.trend,color='red', label='Tendencia')
    plt.title('Componente de Tendencia')
    plt.xlabel('Fecha')
    plt.ylabel('Precio')
    plt.xticks(rotation=45)
    plt.legend(loc='upper left')
    plt.grid(axis='y')  # Cuadrícula horizontal
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))    

    plt.subplot(413)
    plt.plot(decomposition.seasonal,color='blue', label='Estacionalidad')
    plt.title('Componente Estacional')
    plt.xlabel('Fecha')
    plt.ylabel('Efecto Estacional')
    plt.xticks(rotation=45)
    plt.legend(loc='upper left') 
    plt.grid(axis='y')  # Cuadrícula horizontal  
    plt.grid(axis='x')  # Cuadrícula vertical  
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))    
    plt.tight_layout()
    plt.show()


    plt.subplot(414)
    plt.plot(decomposition.resid,color='red', label='Ruido')
    plt.title('Componente de Ruido')
    plt.xlabel('Fecha')
    plt.ylabel('Ruido')
    plt.xticks(rotation=45)
    plt.legend(loc='upper left')
    plt.grid(axis='y')  # Cuadrícula horizontal
    plt.tight_layout()
    plt.show()


    # **Punto 6**: Generar un solo gráfico con las tres líneas (observado, tendencia, estacionalidad)
    plt.figure(figsize=(14, 8))
    plt.plot(decomposition26.observed, color='red', label='Original')
    plt.plot(decomposition26.trend, color='blue', label='Tendencia')
    plt.plot(decomposition26.seasonal, color='green', label='Estacionalidad')
    plt.title('Componentes de la Serie Temporal 26')
    plt.xlabel('Fecha')
    plt.ylabel('Valores')
    plt.xticks(rotation=45)
    plt.legend(loc='upper left')
    plt.grid(axis='y')
    plt.show()


    plt.figure(figsize=(14, 8))
    plt.plot(decomposition52.observed, color='red', label='Original')
    plt.plot(decomposition52.trend, color='blue', label='Tendencia')
    plt.plot(decomposition52.seasonal, color='green', label='Estacionalidad')
    plt.title('Componentes de la Serie Temporal 52')
    plt.xlabel('Fecha')
    plt.ylabel('Valores')
    plt.xticks(rotation=45)
    plt.legend(loc='upper left')
    plt.grid(axis='y')
    plt.show()

