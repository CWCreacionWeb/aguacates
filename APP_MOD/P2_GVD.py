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
from APPModels.APP_FUN import APP_Enunciados,chart


Datos =''    

# --------------------- 2. Gráficos para Visualización de Datos ---------------------
def P21_Grafico_Violin_Volumen_Venta_RegionB():
    APP_Enunciados.getEnunciado("2.1")
    chart.figureConfig(title="Distribución de Ventas por Región (Top 5 Regiones)",xlabel="Región",ylabel="Volumen Total de Ventas")
    sns.violinplot(x='region', y='Total Volume', data=chart.df, hue='region', palette="muted", dodge=False, legend=False)

    APP_Enunciados.getExplicacion("2.1")

def P21_Grafico_Violin_Volumen_Venta_Region(pClasificacionRegion ='', pListaRegiones =''):
    """
1. **Gráfico de Violín de Volumen de Ventas por Región:**
   - **Uso de Datos:** Usa las columnas `Total Volume` y `region`.
   - **Esperado:** Visualiza la distribución de ventas en diferentes regiones.
     - Utiliza la función `violinplot` de `seaborn` para crear gráficos de violín.
     - Configura los ejes para mostrar la relación entre `Total Volume` y `region`.
     - Añade etiquetas y títulos usando `plt.title()` y `plt.xlabel()` para facilitar la interpretación.
    """
    mDbg =P21_Grafico_Violin_Volumen_Venta_Region.__doc__

    mDbg += f"""- **parametros**:  
         - **pClasificacionRegion:** `{pClasificacionRegion}`
         - **pListaRegiones:** `{pListaRegiones}`
    """

    display(Markdown(mDbg))
    plt.figure(figsize=(12, 6))
    #SubData = Datos[Datos['CalRegionGrupo'] == 'GreaterRegion']
    SubData =Datos.copy()
    if pClasificacionRegion != '':
        SubData = SubData[SubData['CalRegionGrupo'] == 'GreaterRegion']


    if pListaRegiones !='':
        SubData = SubData[SubData['region'].isin(pListaRegiones)]

        # Convertir 'Total Volume' a millones
    SubData['Total Volume'] = SubData['Total Volume'] / 1_000_000

    sns.violinplot(x='region', y='Total Volume', data=SubData)
    #sns.violinplot(x=Datos['region'],y=Datos['Total Volume'])
    plt.title("Distribución del Volumen Total de Ventas por Región")
    plt.xlabel("Región")
    plt.ylabel("Volumen Total (millones)")
    plt.xticks(rotation=90)
    plt.show()

def P22_Boxplot_Comparativo_Precios_Entre_Años():
    """
2. **Boxplot Comparativo de Precios entre Años:**
   - **Uso de Datos:** Usa las columnas `AveragePrice` y `year`.
   - **Esperado:** Genera boxplots para comparar la distribución de precios.
     - Utiliza `boxplot` de `seaborn` para crear boxplots que comparen `AveragePrice` entre diferentes años.
     - Asegúrate de que cada boxplot represente un año diferente.
     - Incluye etiquetas y títulos descriptivos usando `plt.title()`.
    """
    mDbg =P22_Boxplot_Comparativo_Precios_Entre_Años.__doc__


    display(Markdown(mDbg))

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='CalYear', y='AveragePrice', data=Datos)
    plt.title("Distribución de Precios Promedios entre Años")
    plt.xlabel("Año")
    plt.ylabel("Precio Promedio")
    plt.show()

def P23_Histograma_Volumen_Total_Ventas(pbins=30, pLog=False):
    """
3. **Histograma de Volumen Total de Ventas:**
   - **Uso de Datos:** Usa la columna `Total Volume`.
   - **Esperado:** Crea un histograma para mostrar la distribución del volumen total de ventas.
     - Utiliza `hist()` de `matplotlib` para crear el histograma.
     - Ajusta el número de bins para una visualización clara usando el parámetro `bins`.
     - Añade etiquetas y un título que describa lo que se muestra.
    """    
    mDbg =P23_Histograma_Volumen_Total_Ventas.__doc__

    mDbg += f"""- **parametros**:  
         - *pbins:*`{pbins}` Numero de intervalos
         - *pLog:*`{pLog}` Escala logaritmica True/False
    """

    display(Markdown(mDbg))

       #if pTipoEscala=='log':
        # Cambiar la escala del eje y a logarítmica
    #plt.yscale('log')

    plt.figure(figsize=(14, 5))

    # Histograma con escala normal
    plt.subplot(1, 2, 1)
    plt.hist(Datos['Total Volume'], bins=pbins, edgecolor='black',log=False)
    plt.title("Distribución del Volumen Total de Ventas")
    plt.xlabel("Volumen Total")
    plt.ylabel("Frecuencia")

    # Histograma con escala logaritmica
    plt.subplot(1, 2, 2)
    plt.hist(Datos['Total Volume'], bins=pbins, edgecolor='black',log=True)
    plt.title("Distribución del Volumen Total de Ventas")
    plt.xlabel("Volumen Total")
    plt.ylabel("Frecuencia (Escala Logarítmica)")
    
    plt.show()
    plt.tight_layout()
    plt.show()    

def P24_Grafico_Barras_Ventas_Tipo_Bolsa():
    """
4. **Gráfico de Barras de Ventas por Tipo de Bolsa:**
   - **Uso de Datos:** Utiliza las columnas `Total Bags`, `Small Bags`, `Large Bags` y `XLarge Bags`.
   - **Esperado:** Compara las ventas de diferentes tipos de bolsas.
     - Suma los volúmenes de ventas por tipo de bolsa utilizando `sum()`.
     - Crea un gráfico de barras con `plt.bar()` para mostrar las diferencias en ventas.
     - Asegúrate de incluir etiquetas para cada tipo de bolsa.

    """

    mDbg =P24_Grafico_Barras_Ventas_Tipo_Bolsa.__doc__


    display(Markdown(mDbg))

    bags = ['Total Bags','Small Bags', 'Large Bags', 'XLarge Bags']
    total_bags = Datos[bags].sum() / 1000000  # Convertir a millones
    plt.figure(figsize=(8, 5))
    #bars=plt.bar(bags, total_bags, color='skyblue')
    bars=plt.bar(bags, total_bags.values, color='skyblue')
    plt.title("Comparación de Ventas por Tipo de Bolsa")
    plt.xlabel("Tipo de Bolsa")
    plt.ylabel("Ventas Totales (millones)")
    # Añadir los valores debajo de cada barra
    for bar, total in zip(bars, total_bags):
        plt.text(
            bar.get_x() + bar.get_width() / 2,   # Posición en X, centrada
            bar.get_height() - (0.05 * bar.get_height()),  # Posición en Y, ligeramente por debajo de la barra
            f'{total:,.0f}',                     # Formato del texto, redondeado sin decimales
            ha='center', va='top', fontsize=10, color='black'
        )

    plt.show()

def P25_Grafico_Lineas_Precios_Promedios_Año(pAnos=''):
    """
5. **Gráfico de Líneas de Precios Promedios por Año:**
   - **Uso de Datos:** Utiliza las columnas `AveragePrice` y `year`.
   - **Esperado:** Visualiza la tendencia de precios promedio a lo largo de los años.
     - Agrupa los datos por `year` y calcula el promedio de `AveragePrice`.
     - Usa `plt.plot()` para crear un gráfico de líneas que muestre la evolución de precios.
     - Añade un título y etiquetas descriptivas a los ejes usando `plt.title()` y `plt.xlabel()`.    
    """

    mDbg =P25_Grafico_Lineas_Precios_Promedios_Año.__doc__

    mDbg += f"""- **parametros**:  
         - *pAnos:*\t`{[pAnos]}`
    """

    display(Markdown(mDbg))

    DatosF = Datos 
    if pAnos =='':
        DatosF = DatosF
    else:
       DatosF = DatosF[DatosF['CalYear'].isin(pAnos)] 



    avg_price_by_year = DatosF.groupby('CalYear')['AveragePrice'].mean()
    plt.figure(figsize=(10, 6))
    plt.plot(avg_price_by_year.index, avg_price_by_year.values, marker='o')
    plt.title("Tendencia de Precios Promedios por Año")
    plt.xlabel("Año")
    plt.ylabel("Precio Promedio")
    plt.show()

