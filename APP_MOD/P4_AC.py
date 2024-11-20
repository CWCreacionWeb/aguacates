from IPython.display import display, Markdown, HTML
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from APPModels.APP_FUN import APP_Enunciados,chart



# Variable global de datos
Datos = None

# 4.1 Cohortes Basadas en Precios Promedios Trimestrales
def P4_1_CohortesPreciosPromedios():
    APP_Enunciados.getEnunciado("4.1")
    APP_Enunciados.getExplicacion("4.1")

    P4_1_CohortesPreciosPromediosB()

def P4_1_CohortesPreciosPromediosB():
    
    # Agrupación trimestral
    datos_trimestrales = Datos.set_index('CalFecha').groupby(pd.Grouper(freq='Q')).agg({
        'AveragePrice': 'mean',
        'Total Volume': 'sum'
    })

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


def P4_2_CohortesRegionFechaB(regiones, anio):
    APP_Enunciados.getEnunciado("4.2")
    APP_Enunciados.getExplicacion("4.2")

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


# 4.3 Análisis de Cohortes en Función del Tipo de Bolsa
def P4_3_CohortesTipoBolsa(pTipoBolsa=['Total Bags','Small Bags','Large Bags','XLarge Bags'],pTipoEscala='',pPorcentaje='NO'):
    APP_Enunciados.getEnunciado("4.3")
    #APP_Enunciados.getExplicacion("4.3")

    mDbg = f"""- **parametros**:  
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

def P4_3_CohortesTipoBolsaB(pTipoBolsa=['Total Bags','Small Bags','Large Bags','XLarge Bags'],pPorcentaje='NO'):
    APP_Enunciados.getEnunciado("4.3")
    #APP_Enunciados.getExplicacion("4.3")
    mDbg = f"""- **parametros**:  
         - **pTipoBolsa:**`{[pTipoBolsa]}` 
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

    # Crear figura y subgráficos para los tres rangos
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 1, 1]})

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
            # Gráfico para el primer rango (0-4)
            vMax = cohortes_bolsas['Small Bags %'].max()
            vMin = cohortes_bolsas['Small Bags %'].min()
            ax1.set_ylim(vMin, vMax)
            vDes = f'Small Bags % rango {vMax - vMin:.2f}' 
            ax1.plot(cohortes_bolsas['CalFecha'], cohortes_bolsas['Small Bags %'], label=vDes, marker='o', color='orange')
            ax1.axhline(y=vMin, color='black', linestyle='--', linewidth=2)  # Doble grosor y punteada
        if 'Large Bags' in pTipoBolsa:
            # Gráfico para el segundo rango (60-70)
            vMax = cohortes_bolsas['Large Bags %'].max()
            vMin = cohortes_bolsas['Large Bags %'].min()
            ax2.set_ylim(vMin, vMax)
            vDes = f'Large Bags % rango  {vMax - vMin:.2f}' 
            ax2.plot(cohortes_bolsas['CalFecha'], cohortes_bolsas['Large Bags %'], label=vDes, marker='o',color='blue')
             # Línea horizontal punteada
            ax2.axhline(y=vMin, color='black', linestyle='--', linewidth=5)  # Doble grosor y punteada
        if 'XLarge Bags' in pTipoBolsa:
            # Gráfico para el tercer rango (80-90)
            vMax = cohortes_bolsas['XLarge Bags %'].max()
            vMin = cohortes_bolsas['XLarge Bags %'].min()
            ax3.set_ylim(vMin, vMax)
            vDes = f'Large Bags % rango  {vMax - vMin:.2f}' 
            ax3.plot(cohortes_bolsas['CalFecha'], cohortes_bolsas['XLarge Bags %'], label=vDes, marker='o',color='red')
            ax3.axhline(y=vMin, color='black', linestyle='--', linewidth=2)  # Doble grosor y punteada
    ax2.legend(fontsize=20)  # Ajusta 'fontsize' al tamaño deseado
    ax3.grid(True, axis='x')  # Añadir cuadrícula vertical
    ax1.grid(True, axis='x')  # Añadir cuadrícula vertical
    ax2.grid(True, axis='x')  # Añadir cuadrícula vertical
    ax3.grid(True, axis='y')  # Añadir cuadrícula vertical
    ax1.grid(True, axis='y')  # Añadir cuadrícula vertical
    ax2.grid(True, axis='y')  # Añadir cuadrícula vertical
    # Personalizar la apariencia para simular un solo gráfico
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax3.spines['top'].set_visible(False)

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    ax3.legend(loc='upper left')

    # Ocultar etiquetas duplicadas
    ax2.tick_params(labeltop=False)
    ax3.tick_params(labeltop=False)

  # Configurar títulos y etiquetas
    fig.suptitle("Análisis de Ventas por Tipo de Bolsa")
    plt.xlabel("Fecha")
    ax2.set_ylabel("Volumen de Ventas")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)  # Ajusta el espacio entre subplots

    plt.show()

# 4.4 Cohortes de Clientes Basadas en Ventas
def P4_4_CohortesClientesVentas():
    APP_Enunciados.getEnunciado("4.4")
    APP_Enunciados.getExplicacion("4.4")

# 4.5 Evaluación de Retención de Ventas por Cohorte
def P4_5_RetencionVentasCohorte():
    APP_Enunciados.getEnunciado("4.5")
    APP_Enunciados.getExplicacion("4.5")

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
