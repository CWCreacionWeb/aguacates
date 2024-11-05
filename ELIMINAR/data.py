"""
Nombre del archivo: data.py
Autor: Jose Niguidula, Alejandro
Descripción: Archivo que contiene código útil para tranformar datos para
            proyecto de curso Data Scientist de la UOC
Creado: 28/10/2024
Versión: 1.0
Correos: ahenaoa@uoc.edu, amanzano2@uoc.edu, jniguidulae@uoc.edu
"""
from hashlib import algorithms_guaranteed

#importes de librerías externas
import pandas as pd
import os

#importe de codigo interno del proyecto
#...

#cargar datos
current_path = os.getcwd()
aguacate_index = current_path.find("aguacate-aljoan")
if aguacate_index != -1:
        project_root = current_path[:aguacate_index + len("aguacate-aljoan")]
else:
    raise FileNotFoundError("The directory 'aguacate-aljoan' was not found in the path.")
data_path = os.path.join(project_root, 'data', 'avocado.csv')
dataset_avocado_original_df = pd.read_csv(data_path)

region_classification = {
    'Albany': 'City',
    'Atlanta': 'City',
    'BaltimoreWashington': 'Region',
    'Boise': 'City',
    'Boston': 'City',
    'BuffaloRochester': 'Region',
    'California': 'GreaterRegion',
    'Charlotte': 'City',
    'Chicago': 'City',
    'CincinnatiDayton': 'Region',
    'Columbus': 'City',
    'DallasFtWorth': 'Region',
    'Denver': 'City',
    'Detroit': 'City',
    'GrandRapids': 'City',
    'GreatLakes': 'GreaterRegion',
    'HarrisburgScranton': 'Region',
    'HartfordSpringfield': 'Region',
    'Houston': 'City',
    'Indianapolis': 'City',
    'Jacksonville': 'City',
    'LasVegas': 'City',
    'LosAngeles': 'City',
    'Louisville': 'City',
    'MiamiFtLauderdale': 'Region',
    'Midsouth': 'GreaterRegion',
    'Nashville': 'City',
    'NewOrleansMobile': 'Region',
    'NewYork': 'City',
    'Northeast': 'GreaterRegion',
    'NorthernNewEngland': 'Region',
    'Orlando': 'City',
    'Philadelphia': 'City',
    'PhoenixTucson': 'Region',
    'Pittsburgh': 'City',
    'Plains': 'GreaterRegion',
    'Portland': 'City',
    'RaleighGreensboro': 'Region',
    'RichmondNorfolk': 'Region',
    'Roanoke': 'City',
    'Sacramento': 'City',
    'SanDiego': 'City',
    'SanFrancisco': 'City',
    'Seattle': 'City',
    'SouthCarolina': 'State',
    'SouthCentral': 'GreaterRegion',
    'Southeast': 'GreaterRegion',
    'Spokane': 'City',
    'StLouis': 'City',
    'Syracuse': 'City',
    'Tampa': 'City',
    'TotalUS': 'TotalUS',
    'West': 'GreaterRegion',
    'WestTexNewMexico': 'Region'
}

def map_regions(original_data: pd.DataFrame, region_map: dict, guardar: bool = False) -> pd.DataFrame:
    """
    Asigna la clasificación de regiones y ciudades al dataframe original de aguacate
    en una nueva columna region_type

    Parametros:
    - original_data: pd.DataFrame-  Datos originales avocado.csv
    - region_map: Dict[str, str] - El mapping de agrupaciones
    - guardar: Boolean - True or False para guardar nuevo csv o no

    Regresa:
    - pd.DataFrame: Dataframe actualizado de los datos originales
    """

    path_salida = "data/avocado_with_region_types.csv"
    nuevo_aguacate_df = original_data.copy()
    nuevo_aguacate_df['region_type'] = nuevo_aguacate_df['region'].map(region_map)
    if guardar:
        print(f"Guardando archivo .csv en /data/  ...")
        nuevo_aguacate_df.to_csv(path_salida, index=False)
    else:
        pass

    return nuevo_aguacate_df

def obtener_nuevo_avocado()-> pd.DataFrame:
    """
    Función que devuelve dataframe con columna region_type

    Regresa:
    - pd.DataFrame: DataFrame con columna nueva que agrupa regiones para análisis
    """

    nuevo_avocado_df = map_regions(dataset_avocado_original_df, region_classification, guardar=False)
    return nuevo_avocado_df

def imputar_fechas()-> pd.DataFrame:
    """
    Función que imouta por promedio las tres entradas faltantes en el avocado.csv original

    regresa:
    - pd.DataFrame: con las tres fechas para el type organic en WestTexNewMexico
    """

    df = obtener_nuevo_avocado()
    df['Date'] = pd.to_datetime(df['Date'])
    # Fechas y parámetros específicos para imputación
    missing_dates = ['2015-12-06', '2017-06-18', '2017-06-25']
    region = 'WestTexNewMexico'
    avocado_type = 'organic'

    # Iterar sobre las fechas faltantes para imputar valores
    for date in missing_dates:
        # Convertir la fecha a datetime
        date = pd.to_datetime(date)

        # Filtrar las filas previas y posteriores a la fecha faltante
        prev_row = df[(df['Date'] < date) &
                      (df['region'] == region) &
                      (df['type'] == avocado_type)].sort_values(by='Date').iloc[-1]
        next_row = df[(df['Date'] > date) &
                      (df['region'] == region) &
                      (df['type'] == avocado_type)].sort_values(by='Date').iloc[0]

        # Calcular el promedio de los valores numéricos entre las dos fechas
        imputed_values = prev_row.copy()
        for col in df.select_dtypes(include='number').columns:
            imputed_values[col] = (prev_row[col] + next_row[col]) / 2

        # Asignar la fecha, región y tipo específico a la fila imputada
        imputed_values['Date'] = date
        imputed_values['region'] = region
        imputed_values['type'] = avocado_type

        # Añadir la fila imputada al DataFrame
        df = pd.concat([df, pd.DataFrame([imputed_values])], ignore_index=True)

        # Ordenar el DataFrame por fecha para mantener el orden cronológico
        df = df.sort_values(by='Date').reset_index(drop=True)

        return df