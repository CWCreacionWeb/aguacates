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

Datos =''    

# --------------------- 3. Elasticidad del Precio ---------------------
# Función para calcular la elasticidad
def calcular_elasticidad(volumen, precio):
    """
### 3. **Elasticidad del Precio**
**Resumen:** El análisis de elasticidad precio-demanda permite evaluar cómo los cambios en los precios afectan la demanda de aguacates. Comprender la elasticidad puede ayudar a formular estrategias de precios más efectivas.

La fórmula de elasticidad precio-demanda es:

$$
E_d = \frac{\% \text{Cambio en la cantidad demandada}}{\% \text{Cambio en el precio}} = \frac{\Delta Q / Q}{\Delta P / P}
$$    
    """
    mDbg =calcular_elasticidad.__doc__

    mDbg += f"""
**Notas**:

- El año 2018 no está completo, tenemos 3 meses, lo que representa que el volumen es un 25%.
- El incremento de volumen respecto al año anterior es incorrecto y, por tanto, la elasticidad también.
- El incremento de precio medio es relativo a 3 meses, por lo que el valor de la elasticidad es inconsistente.
    - Disminuyen las ventas un 71% y el precio un 11%, resultando en una elasticidad de 6.

Si homogeneizamos los datos comparando solo 3 meses:
   - Las ventas se incrementan un 14% y el precio un 4%, con una elasticidad de 0.35.

    """

    mDbg += f"""
- **parametros**:  
         - *volumen:*
         - *precio:* 
    """

    display(Markdown(mDbg))


    # Calcular el porcentaje de cambio en volumen y precio
    cambio_volumen = volumen.pct_change()
    cambio_precio = precio.pct_change()
    # Calcular elasticidad precio-demanda
    elasticidad = (cambio_volumen / cambio_precio)#.fillna(0)



    return elasticidad, cambio_volumen, cambio_precio


def P3_1_Elasticidad_Precio_Demanda_Año(pListaRegiones =''):
    """
1. **Elasticidad Precio-Demanda por Año:**
   - **Uso de Datos:** Usa las columnas `AveragePrice` y `Total Volume`.
   - **Esperado:** Calcula la elasticidad del precio de la demanda para cada año.
     - Calcula la variación porcentual de `Total Volume` y `AveragePrice` utilizando `pd.pct_change()`.
     - Utiliza la fórmula de elasticidad para determinar la sensibilidad de la demanda respecto al precio.
     - Presenta los resultados en un gráfico de líneas usando `plt.plot()` para mostrar la elasticidad por año.    
    """

    mDbg =P3_1_Elasticidad_Precio_Demanda_Año.__doc__

    mDbg += f"""- **parametros**:  
         - *pListaRegiones:*{pListaRegiones}
    """
    display(Markdown(mDbg))


    print("Calculando Elasticidad Precio-Demanda por Año...")
    # Agrupar datos por año y calcular la elasticidad anual
    #SubDatos = datos['region'] =''
    SubData= Datos
    if(pListaRegiones==''):
        SubData = SubData
    else:
        SubData = SubData[Datos['region'] == 'TotalUS']

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

def P3_2_Elasticidad_RegionesN():
    """
    Calcula la elasticidad del precio de la demanda por región y año
    y muestra un gráfico de barras comparando elasticidades por región.
    
    - Datos: DataFrame que contiene las columnas 'Total Volume', 'AveragePrice', 'region', y 'CalYear'
    """
    global Datos

    Datos_anual = Datos.groupby('region','CalYear').agg({'Total Volume': 'sum', 'AveragePrice': 'mean'}).reset_index()


    Datos_anual['Elasticidad'], Datos_anual['Cambio_Volumen'], Datos_anual['Cambio_Precio'] =  calcular_elasticidad(Datos_anual['Total Volume'], Datos_anual['AveragePrice'])
    DatosN = Datos_anual.copy()
    """
    DatosN = Datos.copy()
    # Calcular el cambio porcentual en 'Total Volume' y 'AveragePrice' por año y región
    DatosN['Cambio_Volumen'] = DatosN.groupby(['region', 'CalYear'])['Total Volume'].pct_change()
    DatosN['Cambio_Precio'] = DatosN.groupby(['region', 'CalYear'])['AveragePrice'].pct_change()
    


    
    # Calcular la elasticidad como cambio de volumen sobre cambio de precio
    DatosN['Elasticidad'] = DatosN['Cambio_Volumen'] / DatosN['Cambio_Precio']
    """
    # Filtrar los datos para eliminar filas con NaN en elasticidad (primera fila de cada grupo)
    print(DatosN)
    Datos_filtrado = DatosN.dropna(subset=['Elasticidad'])
    
    # Agrupar por región y año para obtener la elasticidad promedio anual en cada región
    Elasticidad_por_region = Datos_filtrado.groupby(['region', 'CalYear'])['Elasticidad'].mean().reset_index()
    
    # Graficar elasticidad por región y año
    plt.figure(figsize=(14, 8))
    
    # Crear un gráfico de barras donde cada región tenga una barra por año
    for region in Elasticidad_por_region['region'].unique():
        # Filtrar datos de la región actual
        datos_region = Elasticidad_por_region[Elasticidad_por_region['region'] == region]
        # Crear gráfico de barras para la región actual
        plt.bar(datos_region['CalYear'] + 0.1 * Elasticidad_por_region['region'].unique().tolist().index(region), 
                datos_region['Elasticidad'], 
                width=0.2, 
                label=f'Región: {region}')
    
    # Configuración del gráfico
    plt.title('Elasticidad Precio-Demanda por Región y Año')
    plt.xlabel('Año')
    plt.ylabel('Elasticidad')
    plt.xticks(Elasticidad_por_region['CalYear'].unique())  # Años como etiquetas en el eje x
    plt.legend(title='Región')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Punto 3.3 Elasticidad a Nivel de Tipo de Bolsa
def P3_3_Elasticidad_BolsasA():
    """
    3. **Elasticidad a Nivel de Tipo de Bolsa:**
       - **Uso de Datos:** Usa las columnas `AveragePrice` y `Total Bags`.
       - **Esperado:** Calcula la elasticidad del precio de la demanda específica para cada tipo de bolsa.
         - Suma los volúmenes de ventas por tipo de bolsa utilizando `groupby()` y `sum()`.
         - Calcula la elasticidad para cada tipo y presenta los resultados en un gráfico comparativo usando `plt.bar()`.
    """
    
    mDbg = P3_3_Elasticidad_Bolsas.__doc__
    display(Markdown(mDbg))

    print("Calculando Elasticidad para Cada Tipo de Bolsa...")

    # Sumar volúmenes de cada tipo de bolsa por año y calcular elasticidad
    Datos_bolsas = Datos.groupby('CalYear').agg({'Total Bags': 'sum', 'AveragePrice': 'mean'}).reset_index()
    
    # Calcular cambios porcentuales en el volumen total de bolsas y el precio promedio
    Datos_bolsas['Cambio_Volumen'] = Datos_bolsas['Total Bags'].pct_change()
    Datos_bolsas['Cambio_Precio'] = Datos_bolsas['AveragePrice'].pct_change()
    
    # Calcular la elasticidad: cambio de volumen dividido por cambio de precio
    Datos_bolsas['Elasticidad'] = Datos_bolsas['Cambio_Volumen'] / Datos_bolsas['Cambio_Precio']
    
    # Eliminar filas con valores NaN (debido al cálculo de pct_change en la primera fila)
    Datos_bolsas = Datos_bolsas.dropna(subset=['Elasticidad'])

    # Gráfico comparativo de elasticidad para cada año
    plt.figure(figsize=(10, 6))
    plt.bar(Datos_bolsas['CalYear'].astype(str), Datos_bolsas['Elasticidad'], color='blue')
    plt.title('Elasticidad Precio-Demanda por Tipo de Bolsa')
    plt.xlabel('Año')
    plt.ylabel('Elasticidad')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
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
    """
    3. **Elasticidad a Nivel de Tipo de Bolsa:**
       - **Uso de Datos:** Usa las columnas `AveragePrice`, `Small Bags`, `Large Bags`, `XLarge Bags`.
       - **Esperado:** Calcula la elasticidad del precio de la demanda específica para cada tipo de bolsa.
         - Suma los volúmenes de ventas por tipo de bolsa utilizando `groupby()` y `sum()`.
         - Calcula la elasticidad para cada tipo y presenta los resultados en un gráfico comparativo usando `plt.bar()`.
    """
    print("Calculando Elasticidad para Cada Tipo de Bolsa...")
    global Datos
    # Agrupar y sumar los volúmenes de cada tipo de bolsa por año y calcular el precio promedio
    Datos_bolsas = Datos.groupby('CalYear').agg({
        'AveragePrice': 'mean', 
        'Small Bags': 'sum', 
        'Large Bags': 'sum', 
        'XLarge Bags': 'sum'
    }).reset_index()

    # Calcular la elasticidad para cada tipo de bolsa
    Datos_bolsas['Elasticidad_Small'] = calcular_elasticidadB(Datos_bolsas['Small Bags'], Datos_bolsas['AveragePrice'])
    Datos_bolsas['Elasticidad_Large'] = calcular_elasticidadB(Datos_bolsas['Large Bags'], Datos_bolsas['AveragePrice'])
    Datos_bolsas['Elasticidad_XLarge'] = calcular_elasticidadB(Datos_bolsas['XLarge Bags'], Datos_bolsas['AveragePrice'])

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
    """
    4. **Análisis de Elasticidad Comparativa entre Orgánicos y Convencionales:**
       - **Uso de Datos:** Usa las columnas `AveragePrice`, `Total Volume` y `type`.
       - **Esperado:** Compara la elasticidad de la demanda entre aguacates orgánicos y convencionales.
         - Agrupa los datos por `type` y calcula la elasticidad utilizando `pd.pct_change()`.
         - Presenta un gráfico que muestre la diferencia en elasticidad entre los dos tipos usando `plt.plot()`
    """
    
    mDbg = P3_4_Elasticidad_Tipo.__doc__
    display(Markdown(mDbg))

    print("Comparando Elasticidad entre Aguacates Orgánicos y Convencionales...")

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
    """
5. **Análisis de la Elasticidad Precios-Ventas:**
   - **Uso de Datos:** Usa las columnas `AveragePrice` y `Total Volume`.
   - **Esperado:** Examina cómo las variaciones en `AveragePrice` afectan a `Total Volume`.
     - Realiza un análisis de la relación entre estas dos variables calculando la elasticidad.
     - Presenta un gráfico de dispersión que muestre la relación y discute la tendencia observada utilizando `plt.scatter()` y `plt.plot()`    
    """
    mDbg =P3_5_Elasticidad_Precio_Ventas.__doc__

    display(Markdown(mDbg))

    print("Analizando Elasticidad entre Precios y Ventas Totales...")
    # Calcular elasticidad entre precio promedio y volumen total
    elasticidad = calcular_elasticidad(Datos['Total Volume'], Datos['AveragePrice'])
    Datos['Elasticidad_Precio_Ventas'] = elasticidad
    # Gráfico de dispersión de la relación entre precio y volumen
    plt.figure(figsize=(10, 6))
    plt.scatter(Datos['AveragePrice'], Datos['Total Volume']/1000, alpha=0.5, color='purple')
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

