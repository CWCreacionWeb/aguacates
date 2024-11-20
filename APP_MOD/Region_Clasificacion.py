from IPython.display import display, Markdown, HTML
import pandas as pd
import APPModels.APP_FUN as app_fun  # Importa el módulo completo
from APPModels.APP_FUN import APP_Enunciados
# Array que representa la clasificación de regiones
Region_Segmentacion = [
    ['Albany', 'City'],
    ['Atlanta', 'City'],
    ['BaltimoreWashington', 'Region'],
    ['Boise', 'City'],
    ['Boston', 'City'],
    ['BuffaloRochester', 'Region'],
    ['California', 'GreaterRegion'],
    ['Charlotte', 'City'],
    ['Chicago', 'City'],
    ['CincinnatiDayton', 'Region'],
    ['Columbus', 'City'],
    ['DallasFtWorth', 'Region'],
    ['Denver', 'City'],
    ['Detroit', 'City'],
    ['GrandRapids', 'City'],
    ['GreatLakes', 'GreaterRegion'],
    ['HarrisburgScranton', 'Region'],
    ['HartfordSpringfield', 'Region'],
    ['Houston', 'City'],
    ['Indianapolis', 'City'],
    ['Jacksonville', 'City'],
    ['LasVegas', 'City'],
    ['LosAngeles', 'City'],
    ['Louisville', 'City'],
    ['MiamiFtLauderdale', 'Region'],
    ['Midsouth', 'GreaterRegion'],
    ['Nashville', 'City'],
    ['NewOrleansMobile', 'Region'],
    ['NewYork', 'City'],
    ['Northeast', 'GreaterRegion'],
    ['NorthernNewEngland', 'Region'],
    ['Orlando', 'City'],
    ['Philadelphia', 'City'],
    ['PhoenixTucson', 'Region'],
    ['Pittsburgh', 'City'],
    ['Plains', 'GreaterRegion'],
    ['Portland', 'City'],
    ['RaleighGreensboro', 'Region'],
    ['RichmondNorfolk', 'Region'],
    ['Roanoke', 'City'],
    ['Sacramento', 'City'],
    ['SanDiego', 'City'],
    ['SanFrancisco', 'City'],
    ['Seattle', 'City'],
    ['SouthCarolina', 'State'],
    ['SouthCentral', 'GreaterRegion'],
    ['Southeast', 'GreaterRegion'],
    ['Spokane', 'City'],
    ['StLouis', 'City'],
    ['Syracuse', 'City'],
    ['Tampa', 'City'],
    ['TotalUS', 'TotalUS'],
    ['West', 'GreaterRegion'],
    ['WestTexNewMexico', 'Region']
]

# Definir función para categorizar fechas en estaciones
def get_season(date):
    if date.month in [12, 1, 2]:
        return 'Winter'
    elif date.month in [3, 4, 5]:
        return 'Spring'
    elif date.month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autoum'
    

Estacion_Segmentacion = {
    1: 'Invierno',
    2: 'Invierno',
    3: 'Primavera',
    4: 'Primavera',
    5: 'Primavera',
    6: 'Verano',
    7: 'Verano',
    8: 'Verano',
    9: 'Otoño',
    10: 'Otoño',
    11: 'Otoño',
    12: 'Invierno'
}

Lista_CalRegionGrupo =''


def PreparacionDatosSegmentacion(pDfDatos):
    """

- **PreparacionDatosSegmentacion** Añade las siguientes columnas de Segmentación a la tabla: 
    - **CalRegionGrupo:** Agrupación de region en `City,Region,GreaterRegion,TotalUS`
    - **CalEstacion:** Estación del año para ese mes, `Verano,Otoño etc`
    """

    mDbg = PreparacionDatosSegmentacion.__doc__

   

    # Convertir el array a un DataFrame de pandas
    df_Segmentacion = pd.DataFrame(Region_Segmentacion, columns=['region', 'Segmento'])
    Lista_CalRegionGrupo= df_Segmentacion['Segmento'].unique()
    # Paso 3: Crear un diccionario de mapeo de clasificaciones
    Map_Segmentacion = pd.Series(df_Segmentacion['Segmento'].values, index=df_Segmentacion['region']).to_dict()

    # Paso 4: Usar map para agregar la nueva columna 'clasificacionNueva'
    pDfDatos['CalRegionGrupo'] = pDfDatos['region'].map(Map_Segmentacion)
    pDfDatos['CalEstacion'] = pDfDatos['CalMonth'].map(Estacion_Segmentacion)
    if APP_Enunciados.MostrarEnunciado ==True:
        display(Markdown(mDbg))





    
def PreparacionDatosClasificacionVolumen(pDfDatos):
    """
- **PreparacionDatosClasificacionVolumen**  A partir del volumen, calcula el peso de cada region
    - **CalRegion_Total_Volume:** Total Volumen de la región
    - **CalRegion_Porcentaje:** Porcentaje sobre el total
    - **CalRegion_Acumulado_Total_Volume:** Acumulado a efectos de ordenación
    - **CalRegion_Acumulado_Porcentaje:** Acumulado a efectos de ordenación

De este dataFrame obtenido, se desnormaliza y añade a los datos estos campos.

    """


    mDbg = PreparacionDatosClasificacionVolumen.__doc__

   # Agrupar por 'region' y sumar el 'Total Volume' por región
    total_volumen_por_region = pDfDatos.groupby('region')['Total Volume'].sum().reset_index()
    
    # Calcular el total de 'Total Volume' de todas las regiones
    total_volumen = total_volumen_por_region['Total Volume'].sum()
    
    # Calcular el porcentaje de 'Total Volume' por región respecto al total
    total_volumen_por_region['CalRegion_Porcentaje'] = (total_volumen_por_region['Total Volume'] / total_volumen) * 100
    
    # Ordenar las regiones por 'Total Volume' de mayor a menor
    total_volumen_por_region = total_volumen_por_region.sort_values(by='Total Volume', ascending=False).reset_index(drop=True)
    
    # Calcular el acumulado de 'Total Volume' y de porcentaje
    total_volumen_por_region['CalRegion_Acumulado_Total_Volume'] = total_volumen_por_region['Total Volume'].cumsum()
    total_volumen_por_region['CalRegion_Acumulado_Porcentaje'] = total_volumen_por_region['CalRegion_Porcentaje'].cumsum()
    
    # Renombrar la columna de 'Total Volume' para usar prefijo
    total_volumen_por_region.rename(columns={'Total Volume': 'CalRegion_Total_Volume'}, inplace=True)
    
    
  # Fusionar manualmente para actualizar el dataframe original
    pDfDatos['CalRegion_Total_Volume'] = pDfDatos['region'].map(total_volumen_por_region.set_index('region')['CalRegion_Total_Volume'])
    pDfDatos['CalRegion_Porcentaje'] = pDfDatos['region'].map(total_volumen_por_region.set_index('region')['CalRegion_Porcentaje'])
    pDfDatos['CalRegion_Acumulado_Total_Volume'] = pDfDatos['region'].map(total_volumen_por_region.set_index('region')['CalRegion_Acumulado_Total_Volume'])
    pDfDatos['CalRegion_Acumulado_Porcentaje'] = pDfDatos['region'].map(total_volumen_por_region.set_index('region')['CalRegion_Acumulado_Porcentaje'])

    # Fusionar los resultados con el dataframe original
    #pDfDatos = pDfDatos.merge(total_volumen_por_region, on='region', how='left', inplace=True)    
    if APP_Enunciados.MostrarEnunciado ==True:
        display(Markdown(mDbg))

    return total_volumen_por_region

# Ejemplo de uso
# df = pd.read_csv('tu_archivo.csv')
# resultado = clasificar_regiones_por_volumen(df)
# print(resultado)
