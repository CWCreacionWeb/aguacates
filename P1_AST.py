
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime
import numpy as np
import timeit
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import math
import pandas as pd
import UTL_Combo as UTL_CBO
#from IPython.display import display
import ULT_FUNC as M_UF
from ipywidgets import widgets, VBox, HBox, Output, Button

DatosORG =None
Datos = None
Lista_CalRegionGrupo =None
# Creamos un widget de salida para mostrar los resultados
salida = Output()

def Btn_Ejecutar(seleccion):
    print("Función personalizada. Selección realizada:", seleccion)

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

def P1_1_Inicio():

    vLista = M_UF.Lista_Atributo(Datos,'region')
    print(vLista)
    print ('P1_1_Inicio')


    vCBO_region = UTL_CBO.Widget_lst(vLista,'Regiones','BTN Regiones',Btn_Ejecutar)
    vCBO_region.mostrar()


    vLista = M_UF.Lista_Atributo(Datos,'CalRegionGrupo')
    print(vLista)

    vCBO_CalRegionGrupo = UTL_CBO.Widget_lst(vLista,'CalRegionGrupo','BTN CalRegionGrupo',Btn_EjecutarRG)
    vCBO_CalRegionGrupo.mostrar()

    vLista  = ["Todos", "Region Grupo", "Region"]
    vCBO_RegionNivel = UTL_CBO.Widget_lst(vLista,'RegionNivel','BTN RegionNivel',Btn_EjecutarRN)
    vCBO_RegionNivel.mostrar()


    

# --------------------- 1. Análisis de Series Temporales ---------------------

# P1.1_DescomposicionSerieTemporal
def P1_1_DescomposicionSerieTemporal(pPeriodo=52,pCampo='AveragePrice'):
    mDbg=''    
    mDbg +=f'**************************************************************\n'
    mDbg +=f'Parametro pPeriodo:{pPeriodo}\n'
    mDbg +=f'Parametro pCampo:{pCampo}\n'
    mDbg +=f'--------------------------------------------------------------\n'
    mDbg +=f'1.Descomposición de Series Temporales de Precios:\n'
    mDbg +=f'  <strong>Uso de Datos</strong>: Usa la columna AveragePrice y Date.\n'
    mDbg +=f'  Esperado: Utiliza la función seasonal_decompose de la librería statsmodels\n'
    mDbg +=f'  para descomponer la serie temporal de precios en componentes de \n'
    mDbg +=f'  tendencia, estacionalidad y ruido.\n'
    mDbg +=f'Convierte Date a tipo datetime usando pd.to_datetime().\n'
    mDbg +=f'Agrupa los datos por Date y calcula el promedio de AveragePrice utilizando groupby() si es necesario.\n'
    mDbg +=f'Visualiza los componentes descompuestos usando matplotlib para cada uno de ellos.\n'
    mDbg +=f'**************************************************************\n'
    print(mDbg)


    precios = Datos.groupby('CalFecha')[pCampo].mean()
    decomposicion = seasonal_decompose(precios, model='additive', period=pPeriodo)
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
    decomposicion.observed.plot(ax=ax1, title=f'{pCampo} Promedio Observado',xlabel='')
    decomposicion.trend.plot(ax=ax2, title="Tendencia",xlabel='')
    decomposicion.seasonal.plot(ax=ax3, title="Estacionalidad",xlabel='')
    decomposicion.resid.plot(ax=ax4, title="Ruido",xlabel='')
    plt.xlabel("Fecha")
    plt.ylabel("Precio Promedio")

    plt.tight_layout()
    plt.show()

# P1.2_EstacionalidadPorRegion
def P1_2_EstacionalidadPorRegion():
    plt.figure(figsize=(20, 6))
    for region, data in Datos.groupby('region'):
        if region in['Albany','Boston']:
            precios_region = data.groupby('CalFecha')['AveragePrice'].mean()
            #precios_region = Datos.groupby('CalFecha').agg({'AveragePrice':'mean','Total Volume':'mean'}).reset_index()
            plt.plot(precios_region.index, precios_region.values, label=region)
    
    
    plt.xlabel("Fecha")
    plt.ylabel("Precio Promedio")
    plt.title("Estacionalidad del Precio de Aguacates por Región")
    plt.legend()
    plt.show()


def P1A_AnalisisEstacionalidadRegion(region='Albany'):
    """
    Análisis de estacionalidad por región: Precio promedio y volumen total a lo largo del tiempo.
    
    Usa las columnas 'AveragePrice', 'Fecha' y 'Total Volume'.
    Agrupa por 'Region' y 'Fecha' y calcula el promedio de precio y volumen.
    Representa gráficamente las tendencias para una región específica.
    """
    global Datos
    
    # Filtrar datos de la región específica
    datos_region = Datos[Datos['region'] == region]
    
    # Agrupar por fecha y calcular la media de 'AveragePrice' y 'Total Volume'
    datos_agrupados = datos_region.groupby('Fecha').agg({
        'AveragePrice': 'mean',
        'Total Volume': 'mean'
    }).reset_index()
    
    # Crear la figura y el primer eje para 'AveragePrice'
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Configuración del eje para 'AveragePrice'
    ax1.set_title(f"Estacionalidad de Precio Promedio y Volumen Total en la Región: {region}")
    ax1.set_xlabel("Fecha")
    ax1.set_ylabel("Precio Promedio (USD)", color='blue')
    ax1.plot(datos_agrupados['Fecha'], datos_agrupados['AveragePrice'], label='Average Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Crear el segundo eje para 'Total Volume'
    ax2 = ax1.twinx()
    ax2.set_ylabel("Volumen Total", color='green')
    ax2.plot(datos_agrupados['Fecha'], datos_agrupados['Total Volume'], label='Total Volume', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Mostrar la gráfica
    fig.tight_layout()
    plt.show()

def P1B_AnalisisEstacionalidadRegion():
    """
    Análisis de estacionalidad por región: Precio promedio y volumen total a lo largo del tiempo.
    
    Usa las columnas 'AveragePrice', 'Fecha' y 'Total Volume'.
    Agrupa por 'Region' y 'Fecha' y calcula el promedio de precio y volumen.
    Representa gráficamente las tendencias para una región específica.
    """
    global Datos
    
    # Filtrar datos de la región específica
    datos_region = Datos
    
    # Agrupar por fecha y calcular la media de 'AveragePrice' y 'Total Volume'
    datos_agrupados = datos_region.groupby('Fecha').agg({
        'AveragePrice': 'mean',
        'Total Volume': 'mean'
    }).reset_index()
    
    # Crear la figura y el primer eje para 'AveragePrice'
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Configuración del eje para 'AveragePrice'
    ax1.set_title(f"Estacionalidad de Precio Promedio y Volumen Total:")
    ax1.set_xlabel("Fecha")
    ax1.set_ylabel("Precio Promedio (USD)", color='blue')
    ax1.plot(datos_agrupados['Fecha'], datos_agrupados['AveragePrice'], label='Average Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Crear el segundo eje para 'Total Volume'
    ax2 = ax1.twinx()
    ax2.set_ylabel("Volumen Total", color='green')
    ax2.plot(datos_agrupados['Fecha'], datos_agrupados['Total Volume'], label='Total Volume', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Mostrar la gráfica
    fig.tight_layout()
    plt.show()


# Pº_ComparacionPreciosPromedioMensuales
def P1_3_ComparacionPreciosPromedioMensuales(pCampo='AveragePrice'):
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
    plt.figure(figsize=(20, 6))
    volumen_total = Datos.groupby('CalFecha')[pCampo].sum()
    
    plt.plot(volumen_total.index, volumen_total.values, label=f"{pCampo}")
    plt.xlabel("Fecha")
    plt.ylabel(f"{pCampo}")
    plt.title(f"Tendencia {pCampo} de Aguacates a lo Largo del Tiempo")
    plt.legend()
    plt.show()

# P1.5_AnalisisCambiosPreciosAnuales
def P1_5_AnalisisCambiosPreciosAnuales(pClasificacion ='Cyti',pCampo='AveragePrice',pxCampo='CalYear'):
    plt.figure(figsize=(20, 6))
        # Filtrar datos de la región específica
    if pClasificacion =='':
        DatosF = Datos
    else:
        DatosF = Datos[Datos['CalRegionGrupo'] == pClasificacion]
    print(len(DatosF))
    if pCampo =='AveragePrice':
        precios_anuales = DatosF.groupby(pxCampo)[pCampo].mean()
    elif pCampo =='Total Volume':
        precios_anuales = DatosF.groupby(pxCampo)[pCampo].sum()
    
    barras=plt.bar(precios_anuales.index, precios_anuales.values, color='skyblue', label=f"{pCampo} Anual")

    # Añadir etiquetas a las barras
    for barra in barras:
        yval = barra.get_height()
        vValorFormateado =  "{:,.2f}".format(yval)#.replace(",", ".").replace(".", ",", 1)
        plt.text(barra.get_x() + barra.get_width() / 2, yval, vValorFormateado, ha='center', va='bottom')
 

    plt.grid(axis='x')  # Cuadrícula vertical
    #plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%y'))    
    plt.xticks(rotation=45)

    plt.xlabel(f'{pxCampo}')
    plt.ylabel(f'{pCampo}')
    plt.title(f"Análisis de Cambios en {pCampo} Anuales:{pClasificacion}")
    plt.legend()
    plt.show()


