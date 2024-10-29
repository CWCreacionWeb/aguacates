import pandas as pd
import matplotlib.pyplot as plt

# Variable global de datos
Datos = None

# 4.1 Cohortes Basadas en Precios Promedios Trimestrales
def P4_1_CohortesPreciosPromedios():
    P4_1_CohortesPreciosPromediosA()
    P4_1_CohortesPreciosPromediosB()

def P4_1_CohortesPreciosPromediosB():
    """
    Resumen: El análisis de cohortes permite observar el cambio en precios y volúmenes a lo largo del tiempo
    agrupando los datos por trimestres. Aquí se agrupan los datos por fecha trimestral, calculando el promedio de 
    'AveragePrice' y el total de 'Total Volume' en cada cohorte trimestral.
    """
    print("Análisis de Cohortes Basadas en Precios Promedios Trimestrales:")
    
    # Agrupación trimestral
    datos_trimestrales = Datos.set_index('CalFecha').groupby(pd.Grouper(freq='Q')).agg({
        'AveragePrice': 'mean',
        'Total Volume': 'sum'
    })
    print("Datos trimestrales agrupados y calculados por promedio de precio y volumen total.")

    # Años y cuartiles para cada año
    years = datos_trimestrales.index.year.unique()
    
    # Crear posiciones y etiquetas para el eje x
    xticks_labels = []
    xticks_positions = []
    
    for year in years:
        # Posiciones para cada cuartil
        for i in range(4):
            xticks_positions.append(pd.Timestamp(f"{year}-{(i + 1) * 3}-01"))  # Primer día de cada cuatrimestre
            xticks_labels.append('Q' + str(i + 1))  # Añadir cuartil

    # Crear etiquetas de años para los cuartiles
    year_labels = []
    for year in years:
        year_labels.extend([''] * 4)  # Espacio en blanco para cada cuartil
        year_labels[-4] = str(year)  # Colocar el año en la posición del último cuartil

    # Gráfico 1: Precio Promedio Trimestral
    plt.figure(figsize=(12, 6))
    plt.plot(datos_trimestrales.index, datos_trimestrales['AveragePrice'], label="Precio Promedio", color='blue', marker='o')
    plt.title("Cohortes de Precios Promedios Trimestrales")
    plt.xlabel("Fecha")
    plt.ylabel("Precio Promedio")

    # Ajustar ticks y etiquetas
    plt.xticks(xticks_positions, xticks_labels, rotation=0)  # Solo cuartiles
    plt.grid(axis='y', linestyle='--', color='gray')  # Cuadrícula vertical
    for pos in range(1, len(xticks_positions)):  # Añadir líneas para los cuartiles
        plt.axvline(xticks_positions[pos], color='red', linestyle='--', linewidth=0.5)

    # Añadir segunda línea con los años
    plt.gca().set_xticks(xticks_positions)  # Asegurar que se alineen correctamente
    plt.gca().set_xticklabels(xticks_labels)  # Reemplazar etiquetas

    # Añadir años
    plt.xticks(xticks_positions, [f"{label}\n{year_labels[i]}" for i, label in enumerate(xticks_labels)], rotation=0)

    plt.legend()
    plt.tight_layout()  # Ajusta los márgenes
    plt.show()

    # Gráfico 2: Volumen Total Trimestral
    plt.figure(figsize=(12, 6))
    plt.plot(datos_trimestrales.index, datos_trimestrales['Total Volume'], label="Volumen Total", color='green', marker='o')
    plt.title("Cohortes de Volumen Total Trimestrales")
    plt.xlabel("Fecha")
    plt.ylabel("Volumen Total")

    # Ajustar ticks y etiquetas
    plt.xticks(xticks_positions, xticks_labels, rotation=0)  # Solo cuartiles
    plt.grid(axis='y', linestyle='--', color='gray')  # Cuadrícula vertical

   # Añadir líneas verticales solo en posiciones de años
    for year in years:
        plt.axvline(pd.Timestamp(f"{year}-01-01"), color='red', linestyle='--', linewidth=0.5)  # Línea vertical por cada año


    #for pos in range(1, len(xticks_positions)):  # Añadir líneas para los cuartiles
    #    plt.axvline(xticks_positions[pos], color='red', linestyle='--', linewidth=0.5)

    # Añadir segunda línea con los años
    plt.gca().set_xticks(xticks_positions)  # Asegurar que se alineen correctamente
    plt.gca().set_xticklabels(xticks_labels)  # Reemplazar etiquetas

    # Añadir años
    plt.xticks(xticks_positions, [f"{label}\n{year_labels[i]}" for i, label in enumerate(xticks_labels)], rotation=0)

    plt.legend()
    plt.tight_layout()  # Ajusta los márgenes
    plt.show()


def P4_1_CohortesPreciosPromediosA():
    """
    Resumen: El análisis de cohortes permite observar el cambio en precios y volúmenes a lo largo del tiempo
    agrupando los datos por trimestres. Aquí se agrupan los datos por fecha trimestral, calculando el promedio de 
    'AveragePrice' y el total de 'Total Volume' en cada cohorte trimestral.
    """
    print("Análisis de Cohortes Basadas en Precios Promedios Trimestrales:")
    datos_trimestrales = Datos.set_index('CalFecha').groupby(pd.Grouper(freq='Q')).agg({
        'AveragePrice': 'mean',
        'Total Volume': 'sum'
    })
    print("Datos trimestrales agrupados y calculados por promedio de precio y volumen total.")

    datos_trimestrales.plot(y=['AveragePrice', 'Total Volume'], subplots=True, title="Cohortes de Precios Promedios y Volumen Total Trimestrales")
    plt.show()


def P4_2_CohortesRegionFechaB(regiones, anio):
    """
    Resumen: Analiza las cohortes de precios promedio y volumen total por región y año.
    Esta función agrupa los datos por región y fecha para calcular el promedio de precios
    y el volumen total, permitiendo observar las variaciones entre diferentes regiones.
    
    Parámetros:
    - regiones: Lista de regiones a mostrar.
    - anio: Año a filtrar.
    """
    print(f"Análisis de Cohortes por Región para el año: {anio}.")

    # Filtrar datos por año y regiones
    Datos_filtrados = Datos[(Datos['CalYear'] == anio) & (Datos['region'].isin(regiones))]

    # Agrupación de datos por región y fecha
    cohortes_region_fecha = Datos_filtrados.groupby(['region', 'CalFecha']).agg({
        'AveragePrice': 'mean',
        'Total Volume': 'sum'
    }).reset_index()

    print("Datos agrupados por región y fecha, calculando precios promedios y volúmenes totales.")

    # Visualización: Gráfico de precios promedios
    plt.figure(figsize=(12, 6))
    for region in cohortes_region_fecha['region'].unique():
        region_data = cohortes_region_fecha[cohortes_region_fecha['region'] == region]
        plt.bar(region_data['CalFecha'], region_data['AveragePrice'], label=region, alpha=0.7)

    plt.title("Cohortes de Precios Promedios por Región")
    plt.xlabel("Fecha")
    plt.ylabel("Precio Promedio")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', color='gray')
    plt.legend(title='Regiones')
    plt.tight_layout()
    plt.show()

    # Visualización: Gráfico de volumen total
    plt.figure(figsize=(12, 6))
    for region in cohortes_region_fecha['region'].unique():
        region_data = cohortes_region_fecha[cohortes_region_fecha['region'] == region]
        plt.bar(region_data['CalFecha'], region_data['Total Volume'], label=region, alpha=0.7)

    plt.title("Cohortes de Volumen Total por Región")
    plt.xlabel("Fecha")
    plt.ylabel("Volumen Total")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', color='gray')
    plt.legend(title='Regiones')
    plt.tight_layout()
    plt.show()


def P4_2_CohortesRegionFecha(regiones, anio =''):
    """
    Resumen: Analiza las cohortes de precios promedio y volumen total por región y año.
    Esta función agrupa los datos por región y fecha para calcular el promedio de precios
    y el volumen total, permitiendo observar las variaciones entre diferentes regiones.
    
    Parámetros:
    - regiones: Lista de regiones a mostrar.
    - anio: Año a filtrar.
    """
    print(f"Análisis de Cohortes por Región para el año: {anio}.")

    # Filtrar datos por año y regiones
    if anio =='':
        Datos_filtrados = Datos[ (Datos['region'].isin(regiones))]
    else :
        Datos_filtrados = Datos[(Datos['CalYear'] == anio) & (Datos['region'].isin(regiones))]

    # Agrupación de datos por región y fecha
    cohortes_region_fecha = Datos_filtrados.groupby(['region', 'CalFecha']).agg({
        'AveragePrice': 'mean',
        'Total Volume': 'sum'
    }).reset_index()

    print("Datos agrupados por región y fecha, calculando precios promedios y volúmenes totales.")
    
    # Visualización
    plt.figure(figsize=(12, 6))
    for region in cohortes_region_fecha['region'].unique():
        region_data = cohortes_region_fecha[cohortes_region_fecha['region'] == region]
        plt.plot(region_data['CalFecha'], region_data['AveragePrice'], label=region,alpha=0.7) #, marker='o'


    plt.title("Cohortes de Precios Promedios por Región")
    plt.xlabel("Fecha")
    plt.ylabel("Precio Promedio")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', color='gray')
    plt.legend(title='Regiones')
    plt.tight_layout()
    plt.show()

    # Gráfico de Volumen Total
    plt.figure(figsize=(12, 6))
    for region in cohortes_region_fecha['region'].unique():
        region_data = cohortes_region_fecha[cohortes_region_fecha['region'] == region]
        plt.plot(region_data['CalFecha'], region_data['Total Volume'], label=region)

    plt.title("Cohortes de Volumen Total por Región")
    plt.xlabel("Fecha")
    plt.ylabel("Volumen Total")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', color='gray')
    plt.legend(title='Regiones')
    plt.tight_layout()
    plt.show()
# 4.3 Análisis de Cohortes en Función del Tipo de Bolsa
def P4_3_CohortesTipoBolsa():
    """
    Resumen: Analiza cómo se comportan las diferentes cohortes de ventas según el tipo de bolsa.
    Esta función agrupa los datos por tipo de bolsa y fecha, calculando el volumen total de ventas
    para cada tipo de bolsa y visualizando los resultados.
    """
    print("Análisis de Cohortes en Función del Tipo de Bolsa.")

    # Agrupar los datos por tipo de bolsa y fecha
    cohortes_bolsas = Datos.groupby(['CalFecha']).agg({
        'Total Bags': 'sum',
        'Small Bags': 'sum',
        'Large Bags': 'sum',
        'XLarge Bags': 'sum'
    }).reset_index()

    print("Datos agrupados por fecha y calculados los volúmenes de ventas para cada tipo de bolsa.")

    # Visualización: Gráfico de líneas para el volumen de ventas por tipo de bolsa
    plt.figure(figsize=(12, 6))
    plt.plot(cohortes_bolsas['CalFecha'], cohortes_bolsas['Total Bags'], label='Total Bags', marker='o')
    plt.plot(cohortes_bolsas['CalFecha'], cohortes_bolsas['Small Bags'], label='Small Bags', marker='o')
    plt.plot(cohortes_bolsas['CalFecha'], cohortes_bolsas['Large Bags'], label='Large Bags', marker='o')
    plt.plot(cohortes_bolsas['CalFecha'], cohortes_bolsas['XLarge Bags'], label='XLarge Bags', marker='o')

    plt.title("Análisis de Ventas por Tipo de Bolsa")
    plt.xlabel("Fecha")
    plt.ylabel("Volumen de Ventas")
    plt.xticks(rotation=45)
    plt.grid()
    plt.legend(title='Tipo de Bolsa')
    plt.tight_layout()
    plt.show()

# 4.4 Cohortes de Clientes Basadas en Ventas
def P4_4_CohortesClientesVentas():
    """
    Resumen: Agrupa y analiza clientes según el volumen de compras en distintas regiones.
    Este análisis permite ver el comportamiento de las ventas en diferentes cohortes de clientes.
    """
    print("Análisis de Cohortes de Clientes Basadas en Ventas:")
    cohortes_clientes = Datos.groupby(['region', 'CalFecha']).agg({
        'Total Volume': 'sum'
    }).reset_index()
    print("Datos agrupados por volumen de ventas en diferentes regiones.")
    
    for region in cohortes_clientes['region'].unique():
        region_data = cohortes_clientes[cohortes_clientes['region'] == region]
        region_data.plot(x='CalFecha', y='Total Volume', kind='line', title=f"Volumen de Ventas en {region}")
        plt.figure(figsize=(12, 6))
        plt.show()

# 4.5 Evaluación de Retención de Ventas por Cohorte
def P4_5_RetencionVentasCohorte():
    """
    Resumen: Evaluación de la retención de ventas en cohortes a lo largo de un año.
    Analiza la tasa de retención de ventas mensual en distintas cohortes para evaluar la consistencia del mercado.
    """
    print("Evaluación de Retención de Ventas por Cohorte:")
    Datos['Cohorte_Mes'] = Datos['CalFecha'].dt.to_period('M')
    cohortes_retencion = Datos.groupby(['Cohorte_Mes']).agg({
        'Total Volume': 'sum'
    }).reset_index()
    print("Datos de ventas agrupados por cohorte mensual.")

    cohortes_retencion.plot(x='Cohorte_Mes', y='Total Volume', kind='line', title="Retención de Ventas por Cohorte Mensual")
    plt.show()

# Ejemplo de uso:
# cargar_datos('ruta_al_archivo.csv')
# P4_1_CohortesPreciosPromedios()
# P4_2_CohortesRegionFecha()
# P4_3_CohortesTipoBolsa()
# P4_4_CohortesClientesVentas()
# P4_5_RetencionVentasCohorte()
