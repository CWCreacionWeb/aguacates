#from APPModels.APP_FUN import APP_Enunciados,chart,APP_DatosORG
from IPython.display import display, HTML
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import APPModels.APP_FUN as app_fun  # Importa el módulo completo
from APPModels.APP_FUN import APP_Enunciados
from IPython.display import display, Markdown, HTML
import timeit
from APP_MOD import PG_Clases as PG
from APP_MOD import P1_AST as P1
from APP_MOD import P2_GVD as P2
from APP_MOD import P3_EP as P3
from APP_MOD import P4_AC as P4
from APP_MOD import P5_ACR as P5
from APP_MOD import P105_Predicciones as P105_PRED
from APP_MOD import P99_TRAINING as P99_T
import pandas as pd
from APP_MOD import Region_Clasificacion as RC

mFile ='datos/avocado.csv'
mDbg =''
def InicioDoc():
    """
# Análisis del Conjunto de Datos de Precios de Aguacate

**Conjunto de Datos de Precios de Aguacate**: El conjunto de datos "Precios de Aguacate", obtenido de Kaggle, es un conjunto de datos ampliamente utilizado para proyectos de análisis de datos y aprendizaje automático. Proporciona datos históricos sobre precios y ventas de aguacates en varias regiones de los Estados Unidos. Este conjunto de datos es valioso para entender las tendencias en los precios de los aguacates, los volúmenes de ventas y su relación con diferentes factores.

## Atributos Clave

- **Columnas**: El conjunto de datos incluye varias columnas de información. Algunas de las columnas clave típicamente encontradas en este conjunto de datos incluyen:
    - **Fecha** (`Date`): La fecha de observación.
    - **Precio Promedio** (`AveragePrice`): El precio promedio de los aguacates.
    - **Volumen Total** (`Total Volume`): El volumen total de aguacates vendidos.
    - **4046**: Volumen de aguacates Hass pequeños vendidos.
    - **4225**: Volumen de aguacates Hass grandes vendidos.
    - **4770**: Volumen de aguacates Hass extra grandes vendidos.
    - **Bolsas Totales** (`Total Bags`): Total de bolsas de aguacates vendidas.
    - **Bolsas Pequeñas** (`Small Bags`): Bolsas de aguacates pequeños vendidas.
    - **Bolsas Grandes** (`Large Bags`): Bolsas de aguacates grandes vendidas.
    - **Bolsas Extra Grandes** (`XLarge Bags`): Bolsas de aguacates extra grandes vendidas.
    - **Tipo** (`Type`): El tipo de aguacates, generalmente categorizados como convencionales u orgánicos.
    - **Región** (`Region`): La región o ciudad dentro de los Estados Unidos donde se registraron los datos.

- **Rango de Fechas**: El conjunto de datos abarca un rango de fechas, lo que permite el análisis de series de tiempo. Puedes examinar cómo cambian los precios y ventas de aguacates a lo largo de diferentes estaciones y años.

- **Regiones**: Se proporciona información para varias regiones o ciudades a través de los Estados Unidos, lo que permite el análisis de variaciones de precios y ventas en diferentes mercados.

- **Tipos**: El conjunto de datos distingue entre diferentes tipos de aguacates, como convencionales y orgánicos, lo que puede ser útil para comparar tendencias de precios entre estas categorías.

- **Volumen**: Están disponibles datos sobre el volumen total de aguacates vendidos. Esta métrica de volumen se utiliza a menudo para analizar la demanda del mercado.

- **Precio Promedio**: El conjunto de datos contiene el precio promedio de los aguacates, una métrica fundamental para entender las tendencias de precios.

## Casos de Uso

- Este conjunto de datos se utiliza comúnmente para aprender y practicar el análisis de datos, visualización de datos y modelado de regresión en proyectos de ciencia de datos y aprendizaje automático.

- Sirve como un recurso valioso para entender cómo trabajar con datos del mundo real, extraer conocimientos y tomar decisiones basadas en datos.

---
    """


def Cargar(pFile):
    data = pd.read_csv(pFile)
    return data

def Ejecutar():
    """
        Ejecuta los procesos siguientes
        Carga el Fichero CSV definido
        Ejecuta la conversión del campo Date
        mDbg --> Almacena el detalle del resultado 
    """

    global DatosORG
    global mDbg
    
    DatosORG =Cargar(mFile)
    mDbg =""
    mDbg +=f'**********************************\n'
    mDbg +=f'Cargando fichero :{mFile}\n'
    mDbg +=f'numero Registros :{len(DatosORG)}\n'
    mDbg +=f'numero Columnas :{DatosORG.shape[1]}\n'
    mDbg +=f'**********************************\n'
    print(mDbg)
    


def PreparacionDatos():
    """

- **Añade las siguientes columnas a la tabla**: 
    - **CalFecha:** Convierte el campo dato de un string con formato yyyy-mm-dd 
    - **CalYear:** Componente Year de la fecha
    - **CalMes:** Componente Mes de la fecha  

- **Columnas FECHA formato numerico para los modelos de entrenamiento**: 
    - **Cal_AAAAMM:** 
    - **Cal_AAAA:** 
    - **Cal_MM:** 
    - **Cal_MM:** 
    - **Cal_SS:** 
    - **Cal_DDD:** 
    - **Cal_AAAADDD:** 

- **Columnas NORMALIZADAS para los modelos de entrenamiento**: 
    - **CalNOR_Z_TotalVolume:**  Z-Score Normalization
    - **CalNOR_MM_TotalVolume:**  Min-Max Normalization
    - **Cal_NOR_MM_AAAADDD:**  Min-Max Normalization para DDD entre 1 y 1000

    
    """
    #display(Markdown(PreparacionDatos.__doc__))
    global mDbg
    mDbg =PreparacionDatos.__doc__

    DatosORG['CalFecha']=pd.to_datetime(DatosORG['Date'],errors='coerce',format='%Y-%m-%d') 
    errores_conversion = DatosORG['CalFecha'].isna().sum()
    mDbg +=f"""
**Validaciónes**  
  **errores_conversion CalFecha:** {errores_conversion}"""
    # Añadimos el título en negrita y el texto en la siguiente línea en negrita con tabulación
    # Añadimos el título en negrita y el texto en la siguiente línea con tabulación

    # Extraer año y mes para análisis estacional
    DatosORG['CalYear'] = DatosORG['CalFecha'].dt.year
    DatosORG['CalMonth'] = DatosORG['CalFecha'].dt.month

# Generar las columnas solicitadas
    DatosORG['Cal_AAAAMM'] = DatosORG['CalFecha'].dt.year * 100 + DatosORG['CalFecha'].dt.month
    DatosORG['Cal_AAAA'] = DatosORG['CalFecha'].dt.year 
    DatosORG['Cal_MM'] = DatosORG['CalFecha'].dt.month
    DatosORG['Cal_SS'] = DatosORG['CalFecha'].dt.isocalendar().week  # Para la semana del año
    DatosORG['Cal_DDD'] = DatosORG['CalFecha'].dt.dayofyear
    DatosORG['Cal_AAAADDD'] = DatosORG['CalFecha'].dt.year * 1000 + DatosORG['CalFecha'].dt.dayofyear
    # Inicializar el StandardScaler para Z-score
    scaler = StandardScaler()
    # Normalizar el campo Total Volume
    DatosORG['CalNOR_Z_TotalVolume'] = scaler.fit_transform(DatosORG[['Total Volume']])
    # Inicializar el StandardScaler para Z-score
    scaler = MinMaxScaler()
    # Normalizar el campo Total Volume
    DatosORG['CalNOR_MM_TotalVolume'] = scaler.fit_transform(DatosORG[['Total Volume']])


    display(Markdown(mDbg))  
    mDbg =""


DatosRegionClasificacionVolumen = None


def Inicio():
    global mDbg
    global DatosORG
    global DatosRegionClasificacionVolumen
    global Lista_CalRegionGrupo
    global APP_DatosORG
    tiempo_ejecucion = timeit.timeit(lambda: Ejecutar(), number=1) 
    tiempo_ejecucion*=1000
    mDbg+=f'Tiempo de ejecución ms:{tiempo_ejecucion}'

    PreparacionDatos()
    RC.PreparacionDatosSegmentacion(DatosORG)
    DatosRegionClasificacionVolumen= RC.PreparacionDatosClasificacionVolumen(DatosORG)
    Lista_CalRegionGrupo = RC.Lista_CalRegionGrupo
    print(mDbg)
    #DatosORG = DatosORG[(DatosORG['CalRegion_Acumulado_Porcentaje'] > 99.85) | (DatosORG['CalRegion_Acumulado_Porcentaje'] <52)]
    #DatosORG = DatosORG[(DatosORG['CalRegion_Acumulado_Porcentaje'] > 99.85) | (DatosORG['CalRegion_Acumulado_Porcentaje'] <45)]
    app_fun.APP_DatosORG = DatosORG.copy()
    P1.DatosORG = DatosORG


    P1.Datos = DatosORG.copy()

    P1.Lista_CalRegionGrupo = Lista_CalRegionGrupo
    P2.Datos = DatosORG
    P3.Datos = DatosORG
    P4.Datos = DatosORG
    P5.Datos = DatosORG.copy()


#Inicio()
#P1.plot_average_price_by_region_estacion(DatosORG)
#P5.P5_3_PrediccionesMensuales()
#P5.P5_3_PrediccionesMensualesConModeloTRAINING_TODOS()
#P99_T.P99_1_Modelo_TRAINING_TODOS()
#P99_T.P100_1_Modelo_TRAINING_Mod(True)
#P1.P1_1_DescomposicionSerieTemporal()
#P1.P1_2_EstacionalidadPorRegion()
#P1.P1_3_ComparacionPreciosPromedioMensuales()
#P105_PRED.P100_1_Modelo_TRAINING_Mod(pNameModelo='LinearRegression',pFechaReal='2018-10-01')
def DOC():
    APP_Enunciados.getEnunciado('0')
    