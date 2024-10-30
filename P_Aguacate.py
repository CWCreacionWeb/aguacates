import timeit
import P1_AST as P1
import P2_GVD as P2
import P3_EP as P3
import P4_AC as P4
import P5_ACR as P5
import pandas as pd

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

    global Datos
    global mDbg
    
    Datos =Cargar(mFile)
    mDbg =""
    mDbg +=f'**********************************\n'
    mDbg +=f'Cargando fichero :{mFile}\n'
    mDbg +=f'numero Registros :{len(Datos)}\n'
    mDbg +=f'numero Columnas :{Datos.shape[1]}\n'
    mDbg +=f'**********************************\n'
    print(mDbg)
    PreparacionDatos()

def PreparacionDatos():
    """
        Añade las siguientes columnas a la tabla
        CalFecha:Convierte el campo dato de un string con formato yyyy-mm-dd 
        CalYear: Componente Year de la fecha
        CalMes: Componente Mes de la fecha
        mDbg --> Almacena el detalle del resultado de la conversión
    """
    global mDbg
    mDbg +='***********************************************************************\n'
    mDbg +='PreparacionDatos\n'
    mDbg +='Añade las siguientes columnas a la tabla\n'
    mDbg +='   CalFecha:Convierte el campo dato de un string con formato yyyy-mm-dd \n'

    Datos['CalFecha']=pd.to_datetime(Datos['Date'],errors='coerce',format='%Y-%m-%d') 
    errores_conversion = Datos['CalFecha'].isna().sum()
    mDbg +='      Conversion campo Date de string a Datetime formato original YYYY-MM-DD\n'
    mDbg +=f'      errores_conversion --> {errores_conversion}\n'
    # Extraer año y mes para análisis estacional
    Datos['CalYear'] = Datos['CalFecha'].dt.year
    mDbg +='   CalYear: Componente Year de la fecha\n'
    Datos['CalMonth'] = Datos['CalFecha'].dt.month
    mDbg +='   CalMes: Componente Mes de la fecha\n'
    mDbg +='Proceso Finalizado\n'
    mDbg +='***********************************************************************\n'


print('P_Aguacate Ver 0.1\n')
tiempo_ejecucion = timeit.timeit(lambda: Ejecutar(), number=1) 
tiempo_ejecucion*=1000
mDbg+=f'Tiempo de ejecución ms:{tiempo_ejecucion}'

print(mDbg)

P1.Datos = Datos
P2.Datos = Datos
P3.Datos = Datos
P4.Datos = Datos
P5.Datos = Datos

