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
from APP_MOD import P103_Elasticidad as P103_E
from APP_MOD import P99_TRAINING as P99_T
import pandas as pd
from APP_MOD import Region_Clasificacion as RC

mFile ='datos/avocado.csv'
mDbg =''

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
    DatosORG['CalNOR_SS'] = scaler.fit_transform(DatosORG[['Cal_SS']])


    if APP_Enunciados.MostrarEnunciado ==True:
        display(Markdown(mDbg))  
    mDbg =""


DatosRegionClasificacionVolumen = None


def Inicio(pMostrarEnunciado =True):
    global mDbg
    global DatosORG
    global DatosRegionClasificacionVolumen
    global Lista_CalRegionGrupo
    global APP_DatosORG

    APP_Enunciados.MostrarEnunciado = pMostrarEnunciado
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


Inicio(pMostrarEnunciado=False)
#P1.plot_average_price_by_region_estacion(DatosORG)
#P5.P5_3_PrediccionesMensuales()
#P5.P5_3_PrediccionesMensualesConModeloTRAINING_TODOS()
#P99_T.P99_1_Modelo_TRAINING_TODOS()
#P99_T.P100_1_Modelo_TRAINING_Mod(True)
#P1.P1_1_DescomposicionSerieTemporal()
#P1.P1_2_EstacionalidadPorRegion()
#P1.P1_3_ComparacionPreciosPromedioMensuales()
#P105_PRED.P100_1_Modelo_TRAINING_Mod(pNameModelo='LinearRegression',pFechaReal='2018-10-01')
P103_E.Inicio()
#P103_E.P3_1_Elasticidad_Precio_Demanda_01()
#P103_E.mostrar_elasticidad_precio_demanda( agrupacion='MM',dos_escalas=False)
#P103_E.P3_1_Elasticidad_Precio_Demanda_02('MM')
#P103_E.graficar_Data_ElasticidadMensuales_Dispersion()

def DOC():
    APP_Enunciados.getEnunciado('0')
    