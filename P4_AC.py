from IPython.display import display, Markdown, HTML
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Variable global de datos
Datos = None

# 4.1 Cohortes Basadas en Precios Promedios Trimestrales
def P4_1_CohortesPreciosPromedios():
    """

1. **Cohortes Basadas en Precios Promedios Trimestrales:**
   - **Uso de Datos:** Usa las columnas `AveragePrice`, `Total Volume` y `Date`.
   - **Esperado:** Crea cohortes trimestrales y analiza cambios en precios y volúmenes.
     - Agrupa los datos por trimestre usando `pd.Grouper` con `freq='Q'`.
     - Calcula el promedio de `AveragePrice` y suma `Total Volume` para cada cohorte.
     - Visualiza los resultados en un gráfico de líneas que muestre la evolución de las cohortes.    
    """    
    mDbg =P4_1_CohortesPreciosPromedios.__doc__


    display(Markdown(mDbg))

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
2. **Cohortes por Región y Fecha:**
   - **Uso de Datos:** Utiliza las columnas `AveragePrice`, `Total Volume`, `region` y `Date`.
   - **Esperado:** Analiza cómo varían las cohortes de diferentes regiones.
     - Agrupa los datos por `region` y `Date` usando `groupby()`.
     - Calcula el promedio de precios y volumen para cada cohorte.
     - Presenta los resultados en gráficos de barras que muestren comparaciones entre regiones.

    """

    mDbg =P4_2_CohortesRegionFecha.__doc__

    mDbg += f"""- **parametros**:  
         - *regiones:*
         - *anio:* 
    """

    display(Markdown(mDbg))

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
def P4_3_CohortesTipoBolsa(pTipoBolsa=['Total Bags','Small Bags','Large Bags','XLarge Bags'],pTipoEscala='',pPorcentaje='NO'):
    """
3. **Análisis de Cohortes en Función del Tipo de Bolsa:**
   - **Uso de Datos:** Usa las columnas `Total Bags`, `Small Bags`, `Large Bags`, `XLarge Bags` y `Date`.
   - **Esperado:** Examina cómo se comportan las diferentes cohortes según el tipo de bolsa.
     - Agrupa los datos por tipo de bolsa y `Date`.
     - Calcula el volumen de ventas total y muestra los resultados en un gráfico de líneas.

    """

    mDbg =P4_3_CohortesTipoBolsa.__doc__

    mDbg += f"""- **parametros**:  
         - **pTipoBolsa:**`{[pTipoBolsa]}` 
         - **pTipoEscala:**`{[pTipoEscala]}`  **Posibles valores** '' Normal 'log'  Logaritmica
         - **pTipoBolsa:**`{[pPorcentaje]}`   **Posibles valores** SI, NO                
    """

    display(Markdown(mDbg))

    # Agrupar los datos por tipo de bolsa y fecha
    cohortes_bolsas = Datos.groupby(['CalFecha']).agg({
        'Total Bags': 'sum',
        'Small Bags': 'sum',
        'Large Bags': 'sum',
        'XLarge Bags': 'sum'
    }).reset_index()

    # Calcular el porcentaje de cada tipo de bolsa sobre Total Bags
    cohortes_bolsas['Small Bags %'] = (cohortes_bolsas['Small Bags'] / cohortes_bolsas['Total Bags']) * 100 #-70
    cohortes_bolsas['Large Bags %'] = (cohortes_bolsas['Large Bags'] / cohortes_bolsas['Total Bags']) * 100 #- 10
    cohortes_bolsas['XLarge Bags %'] = (cohortes_bolsas['XLarge Bags'] / cohortes_bolsas['Total Bags']) * 100

    # Visualización: Gráfico de líneas para el volumen de ventas por tipo de bolsa
    plt.figure(figsize=(12, 6))
    if pPorcentaje=='NO':
        # Condicional para mostrar solo los tipos de bolsa especificados en pTipoBolsa
        if 'Total Bags' in pTipoBolsa:
            plt.plot(cohortes_bolsas['CalFecha'], cohortes_bolsas['Total Bags'], label='Total Bags', marker='o')
        if 'Small Bags' in pTipoBolsa:
            plt.plot(cohortes_bolsas['CalFecha'], cohortes_bolsas['Small Bags'], label='Small Bags', marker='o')
        if 'Large Bags' in pTipoBolsa:
            plt.plot(cohortes_bolsas['CalFecha'], cohortes_bolsas['Large Bags'], label='Large Bags', marker='o')
        if 'XLarge Bags' in pTipoBolsa:
            plt.plot(cohortes_bolsas['CalFecha'], cohortes_bolsas['XLarge Bags'], label='XLarge Bags', marker='o')
    if pPorcentaje=='SI':        
    # Condicional para mostrar solo los tipos de bolsa especificados en pTipoBolsa
        if 'Small Bags' in pTipoBolsa:
            plt.plot(cohortes_bolsas['CalFecha'], cohortes_bolsas['Small Bags %'], label='Small Bags %', marker='o')
        if 'Large Bags' in pTipoBolsa:
            plt.plot(cohortes_bolsas['CalFecha'], cohortes_bolsas['Large Bags %'], label='Large Bags %', marker='o')
        if 'XLarge Bags' in pTipoBolsa:
            plt.plot(cohortes_bolsas['CalFecha'], cohortes_bolsas['XLarge Bags %'], label='XLarge Bags %', marker='o')

    if pTipoEscala=='log':
        # Cambiar la escala del eje y a logarítmica
        plt.yscale('log')

    # Configurar el formato del eje y para que no use notación científica
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())

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
4. **Cohortes de Clientes Basadas en Ventas:**
   - **Uso de Datos:** Usa las columnas `Total Volume`, `Date` y `region`.
   - **Esperado:** Analiza el comportamiento de las cohortes según el volumen de ventas.
     - Clasifica los clientes según su volumen de compras.
     - Visualiza las cohortes en gráficos de líneas o barras que muestren el comportamiento de compra a lo largo del tiempo.
    """

    mDbg =P4_4_CohortesClientesVentas.__doc__


    display(Markdown(mDbg))



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
5. **Evaluación de Retención de Ventas por Cohorte:**
   - **Uso de Datos:** Usa las columnas `Total Volume` y `Date`.
   - **Esperado:** Estudia cómo se retienen las ventas en cohortes a lo largo de un año.
     - Agrupa los datos por mes y cohortes.
     - Calcula la retención de ventas y visualiza los resultados en un gráfico de líneas que muestre las tasas de retención.
    """

    mDbg =P4_5_RetencionVentasCohorte.__doc__


    display(Markdown(mDbg))


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
