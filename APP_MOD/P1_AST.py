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
import APPModels.APP_FUN as app_fun  # Importa el módulo completo


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

    APP_Enunciados.getExplicacion('1.1')

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


# P1.2_EstacionalidadPorRegion
def P1_2_EstacionalidadPorRegion():
    global Datos
    SubDatos = Datos
    APP_Enunciados.getEnunciado('1.2')
    APP_Enunciados.getExplicacion('1.2')

    plt.figure(figsize=(20, 6))
    # Agrupamos por 'region' y 'CalEstacion', calculamos la media de 'AveragePrice'
    df_grouped = SubDatos.groupby(['region', 'CalYear','CalEstacion'])['AveragePrice'].mean().reset_index()

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
    APP_Enunciados.getEnunciado('1.3')

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


def P1_3_ADD_Correlacion(pHasta = 999999999):
    SubData = Datos.copy()
    SubData = SubData[SubData['Total Volume'] < pHasta ]
    correlation = SubData['Total Volume'].corr(SubData['AveragePrice'])
    print(correlation)
    sns.scatterplot(data=SubData, x='Total Volume', y='AveragePrice')
    plt.title('Relación entre Total Volume y AveragePrice')
    plt.show()



# P1.4_TendenciaVentasALoLargoDelTiempo
def P1_4_TendenciaVentasALoLargoDelTiempo(pCampo='Total Volume'):
    APP_Enunciados.getEnunciado('1.4')

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
    APP_Enunciados.getEnunciado('1.5')

    mDbg = f"""- **parametros**:  
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


