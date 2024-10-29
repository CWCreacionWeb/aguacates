
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

# --------------------- 3. Elasticidad del Precio ---------------------

def P31_Elasticidad_Precio_Demanda_Año():
    Datos['DeltaQ'] = Datos['Total Volume'].pct_change()
    Datos['DeltaP'] = Datos['AveragePrice'].pct_change()
    Datos['Elasticidad'] = (Datos['DeltaQ'] / Datos['Total Volume']) / (Datos['DeltaP'] / Datos['AveragePrice'])
    elasticidad_by_year = Datos.groupby('Year')['Elasticidad'].mean()
    plt.figure(figsize=(10, 6))
    plt.plot(elasticidad_by_year.index, elasticidad_by_year.values, marker='o')
    plt.title("Elasticidad del Precio de la Demanda por Año")
    plt.xlabel("Año")
    plt.ylabel("Elasticidad")
    plt.show()

def P32_Comparacion_Elasticidad_Diferentes_Mercados():
    elasticidad_by_region = Datos.groupby('region')['Elasticidad'].mean()
    plt.figure(figsize=(12, 6))
    plt.bar(elasticidad_by_region.index, elasticidad_by_region.values, color='orange')
    plt.title("Elasticidad del Precio de la Demanda por Región")
    plt.xlabel("Región")
    plt.ylabel("Elasticidad")
    plt.xticks(rotation=90)
    plt.show()

# --------------------- 4. Análisis de Cohortes ---------------------

def P41_Cohortes_Precios_Promedios_Trimestrales():
    Datos.set_index('Fecha', inplace=True)
    quarterly_data = Datos.resample('Q').agg({'AveragePrice': 'mean', 'Total Volume': 'sum'})
    plt.figure(figsize=(10, 6))
    plt.plot(quarterly_data.index, quarterly_data['AveragePrice'], label="Precio Promedio")
    plt.plot(quarterly_data.index, quarterly_data['Total Volume'], label="Volumen Total")
    plt.title("Evolución Trimestral de Precios y Volumen Total")
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.legend()
    plt.show()

# --------------------- 5. Análisis de Correlación y Regresión ---------------------

def P51_Matriz_Correlacion():
    
    #plt.figure(figsize=(10, 8))
    #sns.heatmap(Datos.corr(), annot=True, cmap='coolwarm')
    #plt.title("Matriz de Correlación entre Variables Numéricas")
    #plt.show()
    pass

def P52_Analisis_Dispersion_Variables_Clave():
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='AveragePrice', y='Total Volume', data=Datos)
    sns.regplot(x='AveragePrice', y='Total Volume', data=Datos, scatter=False, color='red')
    plt.title("Análisis de Dispersión entre Precio Promedio y Volumen Total")
    plt.xlabel("Precio Promedio")
    plt.ylabel("Volumen Total")
    plt.show()


def P5_Cambios_Precios_Anuales():
    # 3. Análisis de Cambios en Precios Anuales
    # Crear columna de año para agrupar por año
    #data['Year'] = data['Date'].dt.year

    # Agrupación por año y cálculo del precio promedio anual
    annual_avg_price = Datos.groupby('Year')['AveragePrice'].mean()

    # Visualización de cambios en precios anuales
    plt.figure(figsize=(10, 6))
    plt.bar(annual_avg_price.index, annual_avg_price.values, color='purple', label='Precio Promedio Anual')
    plt.xlabel('Año')
    plt.ylabel('Precio Promedio')
    plt.title('Análisis de Cambios en Precios Anuales')
    plt.legend()
    plt.grid(axis='y')
    plt.show()

def P4_Tendencia_Ventas():
    # Agrupación por fecha y suma del volumen total de ventas
    daily_sales_volume = Datos.groupby('Fecha')['Total Volume'].sum()

    # Visualización de la tendencia de ventas
    plt.figure(figsize=(10, 6))
    plt.plot(daily_sales_volume.index, daily_sales_volume.values, color='g', label='Volumen Total Diario')
    plt.xlabel('Fecha')
    plt.ylabel('Volumen Total de Ventas')
    plt.title('Tendencia de Ventas de Aguacates a lo Largo del Tiempo')
    plt.legend()
    plt.grid()
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
