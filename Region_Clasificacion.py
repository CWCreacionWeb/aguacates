import pandas as pd
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
    'Invierno':[12,1,3],
    'Primavera':[3,4,5],
    'Verano':[6,7,8],
    'Otoño':[9,10,11]
}

Lista_CalRegionGrupo =''
def addClasificacionRegion(pDfDatos):
    # Convertir el array a un DataFrame de pandas
    df_Segmentacion = pd.DataFrame(Region_Segmentacion, columns=['region', 'Segmento'])
    Lista_CalRegionGrupo= df_Segmentacion['Segmento'].unique()
    # Paso 3: Crear un diccionario de mapeo de clasificaciones
    Map_Segmentacion = pd.Series(df_Segmentacion['Segmento'].values, index=df_Segmentacion['region']).to_dict()

    # Paso 4: Usar map para agregar la nueva columna 'clasificacionNueva'
    pDfDatos['CalRegionGrupo'] = pDfDatos['region'].map(Map_Segmentacion)



