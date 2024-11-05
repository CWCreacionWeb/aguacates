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
import APP_MOD.UTL_Combo as UTL_CBO
#from IPython.display import display
from ipywidgets import widgets, VBox, HBox, Output, Button
import APP_MOD.ULT_FUNC as M_UF
from APPModels.APP_FUN import APP_Enunciados,chart

DatosORG =None
Datos = None
Lista_CalRegionGrupo =None
# Creamos un widget de salida para mostrar los resultados
salida = Output()

def Btn_Ejecutar(seleccion):
    global Datos
    global DatosORG
    print("Función personalizada. Selección realizada Region:", seleccion)
    Datos = DatosORG[DatosORG['region'].isin(seleccion)]
    mDbg =""
    mDbg +=f'**********************************\n'
    mDbg +=f'Datos\n'
    mDbg +=f'numero Registros :{len(Datos)}\n'
    mDbg +=f'numero Columnas :{Datos.shape[1]}\n'
    mDbg +=f'**********************************\n'
    print(mDbg)

    mDbg =""
    mDbg +=f'**********************************\n'
    mDbg +=f'DatosORG\n'
    mDbg +=f'numero Registros :{len(DatosORG)}\n'
    mDbg +=f'numero Columnas :{DatosORG.shape[1]}\n'
    mDbg +=f'**********************************\n'
    print(mDbg)


def Btn_EjecutarRG(seleccion):
    global Datos
    global DatosORG
    print("Función personalizada. Selección realizada RG:", seleccion)
    Datos = DatosORG[DatosORG['CalRegionGrupo'].isin(seleccion)]
    mDbg =""
    mDbg +=f'**********************************\n'
    mDbg +=f'Datos\n'
    mDbg +=f'numero Registros :{len(Datos)}\n'
    mDbg +=f'numero Columnas :{Datos.shape[1]}\n'
    mDbg +=f'**********************************\n'
    print(mDbg)

    mDbg =""
    mDbg +=f'**********************************\n'
    mDbg +=f'DatosORG\n'
    mDbg +=f'numero Registros :{len(DatosORG)}\n'
    mDbg +=f'numero Columnas :{DatosORG.shape[1]}\n'
    mDbg +=f'**********************************\n'
    print(mDbg)


def Btn_EjecutarRN(seleccion):
    print("Función personalizada. Selección realizada RG:", seleccion)

def P1_CfgListView():
    print( len(Datos))
    vLista = M_UF.Lista_Atributo(Datos,'region')
    print(vLista)
    print ('P1_1_Inicio')


    vCBO_region = UTL_CBO.Widget_lst(vLista,'Regiones','BTN Regiones',Btn_Ejecutar)
    #vCBO_region.mostrar()


    vLista = M_UF.Lista_Atributo(Datos,'CalRegionGrupo')
    print(vLista)

    vCBO_CalRegionGrupo = UTL_CBO.Widget_lst(vLista,'CalRegionGrupo','BTN CalRegionGrupo',Btn_EjecutarRG)
    #vCBO_CalRegionGrupo.mostrar()

    vLista  = ["Todos", "Region Grupo", "Region"]
    vCBO_RegionNivel = UTL_CBO.Widget_lst(vLista,'RegionNivel','BTN RegionNivel',Btn_EjecutarRN)
    #vCBO_RegionNivel.mostrar()

    # Usar HBox para organizar los tres widgets en una misma fila
    #display(HBox([vCBO_region.wLista_widgets, vCBO_CalRegionGrupo.wLista_widgets, vCBO_RegionNivel.wLista_widgets]))

    # Usamos HBox para organizar los widgets horizontalmente
    display(HBox([vCBO_region.mostrar(), vCBO_CalRegionGrupo.mostrar(), vCBO_RegionNivel.mostrar()]))    

# --------------------- 1. Análisis de Series Temporales ---------------------
def DOC():
    APP_Enunciados.getEnunciado('1')
# P1.1_DescomposicionSerieTemporal
def P1_1_DescomposicionSerieTemporal(pPeriodo=52,pCampo='AveragePrice'):
    APP_Enunciados.getEnunciado('1.1')

    mDbg = f"""- **parametros**:  
         - *pPeriodo:*\t`{pPeriodo}`
         - *pCampo:*\t`{pCampo}`
    """


    display(Markdown(mDbg))

        # Mostramos el texto en formato Markdown

    precios = Datos.groupby('CalFecha')[pCampo].mean()

    decomposicion = seasonal_decompose(precios, model='additive', period=pPeriodo)


    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
    decomposicion.observed.plot(ax=ax1, title=f'{pCampo} Promedio Observado',xlabel='')
    decomposicion.trend.plot(ax=ax2, title="Tendencia",xlabel='')
    decomposicion.seasonal.plot(ax=ax3, title="Estacionalidad",xlabel='')
    decomposicion.resid.plot(ax=ax4, title="Ruido",xlabel='')
    plt.xlabel("Fecha")
    plt.ylabel(f"{pCampo} Promedio")

    plt.tight_layout()
    plt.show()


# Función 1: Evolución media de AveragePrice por region y CalFecha
def plot_average_price_by_region_fecha(df):
    # Agrupamos por 'region' y 'CalFecha', calculamos la media de 'AveragePrice'
    df_grouped = df.groupby(['region', 'CalFecha'])['AveragePrice'].mean().reset_index()
    
    # Graficamos
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=df_grouped, x='CalFecha', y='AveragePrice', hue='region', marker='o')
    plt.title('Evolución media de AveragePrice por región y CalFecha')
    plt.xlabel('Fecha')
    plt.ylabel('Precio Promedio')
    plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Función 2: Evolución media de AveragePrice por region y CalEstacion
def plot_average_price_by_region_estacion(df):
    # Convertimos 'CalEstacion' en una categoría con un orden específico
    #estaciones = ['Invierno', 'Primavera', 'Verano', 'Otoño']
    #df['CalEstacion'] = pd.Categorical(df['CalEstacion'], categories=estaciones, ordered=True)
    #print(df[['CalFecha', 'region', 'AveragePrice', 'CalEstacion']])


    # Agrupamos por 'region' y 'CalEstacion', calculamos la media de 'AveragePrice'
    #df_grouped = df.groupby(['region', 'CalYear','CalEstacion'])['AveragePrice'].mean().reset_index()

  # Agrupamos por 'region', 'CalYear' y 'CalEstacion', y calculamos el promedio ponderado de 'AveragePrice'
    df_grouped = df.groupby(['region', 'CalYear', 'CalEstacion']).apply(
        lambda x: (x['AveragePrice'] * x['Total Volume']).sum() / x['Total Volume'].sum()
    ).reset_index(name='WeightedAveragePrice')

 

        # Crear una nueva columna que combine 'CalYear' y 'CalEstacion' para el eje x
    df_grouped['Year_Estacion'] = df_grouped['CalYear'].astype(str) + ' ' + df_grouped['CalEstacion']


    print(df_grouped.head())
    
    # Graficamos
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=df_grouped, x='Year_Estacion', y='WeightedAveragePrice', hue='region', marker='o')
    plt.title('Evolución media de AveragePrice por región y estación')
    plt.xlabel('Estación')
    plt.ylabel('Precio Promedio')
    plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Función 2: Evolución media de AveragePrice por region y CalEstacion
def plot_average_price_by_region_estacionB(df):
    # Convertimos 'CalEstacion' en una categoría con un orden específico
    #estaciones = ['Invierno', 'Primavera', 'Verano', 'Otoño']
    #df['CalEstacion'] = pd.Categorical(df['CalEstacion'], categories=estaciones, ordered=True)
    #print(df[['CalFecha', 'region', 'AveragePrice', 'CalEstacion']])


    # Agrupamos por 'region' y 'CalEstacion', calculamos la media de 'AveragePrice'
    df_grouped = df.groupby(['region', 'CalYear','CalEstacion'])['AveragePrice'].mean().reset_index()

  

        # Crear una nueva columna que combine 'CalYear' y 'CalEstacion' para el eje x
    df_grouped['Year_Estacion'] = df_grouped['CalYear'].astype(str) + ' ' + df_grouped['CalEstacion']


    print(df_grouped.head())
    
    # Graficamos
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=df_grouped, x='Year_Estacion', y='AveragePrice', hue='region', marker='o')
    plt.title('Evolución media de AveragePrice por región y estación')
    plt.xlabel('Estación')
    plt.ylabel('Precio Promedio')
    plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Función 3: Evolución media ponderada de AveragePrice sobre Total Volume por region y CalFecha
def plot_weighted_average_price_by_region_fecha(df):
    # Calculamos la media ponderada de AveragePrice usando Total Volume
    df['Weighted_AveragePrice'] = df['AveragePrice'] * df['Total Volume']
    df_grouped = df.groupby(['region', 'CalYear']).apply(
        lambda x: x['Weighted_AveragePrice'].sum() / x['Total Volume'].sum()).reset_index(name='Weighted_AveragePrice')
    
    # Graficamos
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=df_grouped, x='CalYear', y='Weighted_AveragePrice', hue='region', marker='o')
    plt.title('Evolución media ponderada de AveragePrice por región y CalFecha')
    plt.xlabel('Fecha')
    plt.ylabel('Precio Promedio Ponderado')
    plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# P1.2_EstacionalidadPorRegion
def P1_2_EstacionalidadPorRegion():
    """
2. **Análisis de Estacionalidad por Región:** 
   - **Uso de Datos:** Usa las columnas `AveragePrice`, `Date` y `Total Volume`.
   - **Esperado:** Utiliza gráficos de líneas para visualizar cómo varían los precios de aguacates por región a lo largo de diferentes estaciones del año.
     - Agrupa los datos por `region` y `Date` utilizando `groupby()`.
     - Calcula el promedio de `AveragePrice` para cada región.
     - Representa gráficamente las tendencias utilizando `plt.plot()` de `matplotlib`.    
    """
    global Datos
    global mDbg
    mDbg =P1_2_EstacionalidadPorRegion.__doc__

    display(Markdown(mDbg))

    plt.figure(figsize=(20, 6))
    # Agrupamos por 'region' y 'CalEstacion', calculamos la media de 'AveragePrice'
    df_grouped = Datos.groupby(['region', 'CalYear','CalEstacion'])['AveragePrice'].mean().reset_index()

        # Crear una nueva columna que combine 'CalYear' y 'CalEstacion' para el eje x
    df_grouped['Year_Estacion'] = df_grouped['CalYear'].astype(str) + ' ' + df_grouped['CalEstacion']


    
    
    # Graficamos
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=df_grouped, x='Year_Estacion', y='AveragePrice', hue='region', marker='o')
    plt.title('Evolución media de AveragePrice por región y estación')
    plt.xlabel('Estación')
    plt.ylabel('Precio Promedio')
    plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Pº_ComparacionPreciosPromedioMensuales
def P1_3_ComparacionPreciosPromedioMensuales(pCampo='AveragePrice'):
    """
3. **Comparación de Precios Promedio Mensuales:**
   - **Uso de Datos:** Usa las columnas `AveragePrice` y `Date`.
   - **Esperado:** Calcula y compara los precios promedio mensuales.
     - Agrupa los datos por mes usando `pd.Grouper` con `freq='M'`.
     - Calcula el promedio de `AveragePrice` para cada mes con `mean()`.
     - Visualiza los resultados con un gráfico de líneas usando `plt.plot()`.
    """
    mDbg =P1_3_ComparacionPreciosPromedioMensuales.__doc__

    display(Markdown(mDbg))

    plt.figure(figsize=(20, 6))
    precios_mensuales = Datos.groupby(pd.Grouper(key='CalFecha', freq='M'))[pCampo].mean()
    
    plt.plot(precios_mensuales.index, precios_mensuales.values, label=f"{pCampo} Mensual")

    plt.grid(axis='x')  # Cuadrícula vertical
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))    
    plt.xticks(rotation=45)
    
    plt.xlabel("Fecha")
    plt.ylabel(f'{pCampo}')
    plt.title(f"Comparación de {pCampo} Mensuales")
    plt.legend()
    plt.show()

# P1.4_TendenciaVentasALoLargoDelTiempo
def P1_4_TendenciaVentasALoLargoDelTiempo(pCampo='Total Volume'):
    """
4. **Tendencia de Ventas a lo Largo del Tiempo:**
   - **Uso de Datos:** Usa las columnas `Total Volume` y `Date`.
   - **Esperado:** Analiza cómo varía el volumen total de ventas a lo largo del tiempo.
     - Agrupa los datos por `Date` y suma el `Total Volume` usando `groupby()`.
     - Visualiza los resultados usando un gráfico de líneas con `plt.plot()` para mostrar la tendencia.    
    """    
    mDbg =P1_4_TendenciaVentasALoLargoDelTiempo.__doc__

    display(Markdown(mDbg))

    plt.figure(figsize=(20, 6))
    volumen_total = Datos.groupby('CalFecha')[pCampo].sum()
    
    plt.plot(volumen_total.index, volumen_total.values, label=f"{pCampo}")

    plt.grid(axis='x')  # Cuadrícula vertical
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))    
    plt.xticks(rotation=45)

    plt.xlabel("Fecha")
    plt.ylabel(f"{pCampo}")
    plt.title(f"Tendencia {pCampo} de Aguacates a lo Largo del Tiempo")
    plt.legend()
    plt.show()


def P1_5_AnalisisCambiosPreciosAnuales(pAnos='', pClasificacion ='',pCampo='AveragePrice',pxCampo='CalYear'):
    """
5. **Análisis de Cambios en Precios Anuales:**
   - **Uso de Datos:** Usa las columnas `AveragePrice` y `year`.
   - **Esperado:** Observa las diferencias anuales en los precios promedio.
     - Agrupa los datos por `year` utilizando `groupby()`.
     - Calcula el promedio de `AveragePrice` para cada año.
     - Representa los resultados en un gráfico de barras usando `plt.bar()` que compare los precios de cada año.
    """
    mDbg =P1_5_AnalisisCambiosPreciosAnuales.__doc__

    mDbg += f"""- **parametros**:  
         - *pAnos:*\t`{[pAnos]}`
         - *pClasificacion:*\t`{[pClasificacion]}` City,Region,GreaterRegion,TotalUS
    """

    display(Markdown(mDbg))

    plt.figure(figsize=(20, 6))
        # Filtrar datos de la región específica
    DatosF = Datos 
    if pAnos =='':
        DatosF = DatosF
    else:
       DatosF = DatosF[DatosF['CalYear'].isin(pAnos)] 

    if pClasificacion =='':
        DatosF = DatosF
    else:
        DatosF = DatosF[DatosF['CalRegionGrupo'] == pClasificacion]
    
    if pCampo =='AveragePrice':
        precios_anuales = DatosF.groupby(pxCampo)[pCampo].mean().reset_index()
    elif pCampo =='Total Volume':
        precios_anuales = DatosF.groupby(pxCampo)[pCampo].sum().reset_index()
    
    # Crear el gráfico de líneas
    plt.figure(figsize=(10, 6))
    plt.bar(precios_anuales[pxCampo], precios_anuales[pCampo], color='skyblue')
    #plt.plot(precios_anuales[pxCampo], precios_anuales[pCampo], marker='o', linestyle='-')

    # Añadir título y etiquetas
    plt.title('Evolución de Precios Promedios de Aguacates por Año')

    plt.grid() 
    #plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%y'))    
    plt.xlabel(f'{pxCampo}')
    plt.ylabel(f'{pCampo}')
    plt.title(f"Análisis de Cambios en {pCampo} Anuales:{pClasificacion}")

    # Añadir etiquetas de valor en cada barra
    for i, valor in enumerate(precios_anuales[pCampo]):
        plt.text(precios_anuales[pxCampo].iloc[i], valor, f'{valor:,.2f}', ha='center', va='bottom', fontsize=10)

    # Mostrar el gráfico
    plt.tight_layout()
    plt.show()    
    #plt.legend()
    #plt.show()


def P1_5_AnalisisCambiosPreciosMensuales():
    # Crear columnas de Año y Mes si no existen
    Datos['xYear'] = Datos['CalFecha'].dt.year
    Datos['xMonth'] = Datos['CalFecha'].dt.month

    # Agrupar por Año y Mes y calcular el promedio de AveragePrice
    precios_mensuales = Datos.groupby(['xYear', 'xMonth'])['AveragePrice'].mean().reset_index()

    # Crear el gráfico de líneas
    plt.figure(figsize=(12, 6))

    # Lista de colores para cada año
    colores = plt.cm.viridis_r(precios_mensuales['xYear'].nunique())

    # Graficar una línea para cada año
    for i, year in enumerate(precios_mensuales['xYear'].unique()):
        # Filtrar los datos para el año actual
        datos_anuales = precios_mensuales[precios_mensuales['xYear'] == year]
        # Crear la línea para el año con un color específico y una leyenda
        #plt.plot(datos_anuales['xMonth'], datos_anuales['AveragePrice'], label=year, color=colores[i], marker='o')
        plt.plot(datos_anuales['xMonth'], datos_anuales['AveragePrice'], label=year, marker='o')

    # Configurar etiquetas y título
    plt.title("Cambios Anuales en Precios Promedio de Aguacates por Mes")
    plt.xlabel("Mes")
    plt.ylabel("Precio Promedio")
    plt.xticks(range(1, 13), ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
    plt.legend(title='Año')
    plt.grid()
    plt.tight_layout()

    # Mostrar el gráfico
    plt.show()

def P1_5_AnalisisCambiosPreciosSemanales():
    print('Las semanas están desplazadas, la primera lectura del año corresponde al domingo no al dia 1\n')
    print('la primera semana puede empezar el 7, y los días anteriores se han computado a la semana 53 del anterior\n')
    print('esto junto a que un mes puede tener 4 o 5 semamas, genera ruido en la información\n')
    print('el mes no es una unidad omogénea\n')



    # Crear columnas de Año y Mes si no existen
    Datos['xYear'] = Datos['CalFecha'].dt.year
    #Datos['xWeek'] = Datos['CalFecha'].dt.isocalendar().week
    Datos['xWeek'] = Datos['CalFecha'].dt.dayofyear

    # Agrupar por Año y Mes y calcular el promedio de AveragePrice
    precios_semanales = Datos.groupby(['xYear', 'xWeek'])['AveragePrice'].mean().reset_index()
    precios_semanales=precios_semanales[precios_semanales['xWeek'] <= 90]
    # Crear el gráfico de líneas
    plt.figure(figsize=(12, 6))

    # Lista de colores para cada año
    colores = plt.cm.viridis_r(precios_semanales['xYear'].nunique())

    # Graficar una línea para cada año
    for i, year in enumerate(precios_semanales['xYear'].unique()):
        # Filtrar los datos para el año actual
        datos_anuales = precios_semanales[precios_semanales['xYear'] == year]
        # Crear la línea para el año con un color específico y una leyenda
        #plt.plot(datos_anuales['xMonth'], datos_anuales['AveragePrice'], label=year, color=colores[i], marker='o')
        plt.plot(datos_anuales['xWeek'], datos_anuales['AveragePrice'], label=year, marker='o')

    # Configurar etiquetas y título
    plt.title("Cambios Anuales en Precios Promedio de Aguacates por Mes")
    plt.xlabel("Semana")
    plt.ylabel("Precio Promedio")
    plt.xticks(range(0, 91,7)) 
    plt.legend(title='Año')
    plt.grid()
    plt.tight_layout()

    # Mostrar el gráfico
    plt.show()
