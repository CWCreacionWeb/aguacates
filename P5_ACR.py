import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


Datos =''    



def P5_1_MatrizCorrelacion():
    """
    Genera y visualiza la matriz de correlación entre las variables numéricas del DataFrame.
    """
    print("Generando matriz de correlación...")
    
    correlacion = Datos[['AveragePrice', 'Total Volume', '4046', '4225', '4770', 'Total Bags']].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlacion, annot=True, cmap='coolwarm')
    plt.title('Matriz de Correlación')
    plt.show()
    
    print("Correlaciones significativas encontradas:")
    print(correlacion)

def P5_2_AnalisisDispersión():
    """
    Crea un gráfico de dispersión entre AveragePrice y Total Volume.
    Añade una línea de regresión para visualizar la relación.
    """
    print("Generando gráfico de dispersión...")
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=Datos, x='Total Volume', y='AveragePrice')
    sns.regplot(data=Datos, x='Total Volume', y='AveragePrice', scatter=False, color='red')
    plt.title('Análisis de Dispersión: AveragePrice vs Total Volume')
    plt.show()
    
    print("El gráfico muestra la relación entre AveragePrice y Total Volume.")

def P5_3_PrediccionesMensuales():
    """
    Realiza predicciones mensuales usando datos trimestrales.
    Compara los resultados de las predicciones con los precios reales.
    """
    print("Realizando predicciones mensuales...")
    
    # Agrupación por trimestres y cálculo de promedios
    Datos_trimestrales = Datos.groupby(pd.Grouper(key='CalFecha', freq='Q')).agg({'AveragePrice': 'mean', 'Total Volume': 'mean'}).reset_index()
    Datos_trimestrales['Month'] = Datos_trimestrales['CalFecha'].dt.month

    # Predicciones
    predicciones = []
    for i in range(len(Datos_trimestrales) - 2):
        promedio_precio = (Datos_trimestrales['AveragePrice'][i] + Datos_trimestrales['AveragePrice'][i + 1]) / 2
        predicciones.append(promedio_precio)
    
    # Comparar con los precios reales
    precios_reales = Datos_trimestrales['AveragePrice'][2:].reset_index(drop=True)
    predicciones_df = pd.DataFrame({'Real': precios_reales, 'Predicción': predicciones})

    print("Comparación de precios reales vs predicciones:")
    print(predicciones_df)

def P5_4_PrediccionesTrimestrales():
    """
    Realiza predicciones trimestrales usando los datos agrupados por trimestres.
    Evalúa la precisión de las predicciones.
    """
    print("Realizando predicciones trimestrales...")
    
    # Agrupación de datos trimestrales
    datos_trimestrales = Datos.groupby(pd.Grouper(key='CalFecha', freq='Q')).mean().reset_index()
    
    # Modelo de regresión lineal
    X = np.array(range(len(datos_trimestrales))).reshape(-1, 1)
    y = datos_trimestrales['AveragePrice'].values
    modelo = LinearRegression()
    modelo.fit(X, y)
    
    # Predicción del siguiente trimestre
    siguiente_trimestre = np.array([[len(datos_trimestrales)]])
    prediccion = modelo.predict(siguiente_trimestre)

    print(f"Predicción del precio promedio para el próximo trimestre: {prediccion[0]}")

def P5_5_PrediccionesAnuales():
    """
    Realiza predicciones anuales agrupando los datos por año.
    Evalúa la precisión de las predicciones.
    """
    print("Realizando predicciones anuales...")
    
    # Agrupación de datos anuales
    datos_anuales = Datos.groupby(pd.Grouper(key='CalFecha', freq='Y')).mean().reset_index()
    
    # Modelo de regresión lineal
    X = np.array(range(len(datos_anuales))).reshape(-1, 1)
    y = datos_anuales['AveragePrice'].values
    modelo = LinearRegression()
    modelo.fit(X, y)
    
    # Predicción del próximo año
    siguiente_año = np.array([[len(datos_anuales)]])
    prediccion = modelo.predict(siguiente_año)

    print(f"Predicción del precio promedio para el próximo año: {prediccion[0]}")

def P5_6_ModelosRegresionMultiple():
    """
    Desarrolla modelos de regresión múltiple para predecir AveragePrice.
    Compara su rendimiento.
    """
    print("Desarrollando modelos de regresión múltiple...")
    
    # Seleccionar variables
    X = Datos[['Total Volume', '4046', '4225', '4770', 'Total Bags']]
    y = Datos['AveragePrice']
    
    # Ajustar modelo de regresión
    modelo = LinearRegression()
    modelo.fit(X, y)
    
    # Predicciones
    predicciones = modelo.predict(X)
    
    # Métricas
    mse = mean_squared_error(y, predicciones)
    r2 = r2_score(y, predicciones)
    
    print(f"MSE: {mse}, R²: {r2}")

def P5_7_CoefficientsRegresionMultiple():
    """
    Analiza los coeficientes del modelo de regresión múltiple ajustado.
    """
    print("Analizando coeficientes de regresión múltiple...")
    
    # Usar el último modelo de regresión múltiple
    coeficientes = modelo.coef_
    
    print("Coeficientes de las variables:")
    for i, coef in enumerate(coeficientes):
        print(f"Variable {X.columns[i]}: {coef}")

def P5_8_VolumenVentas():
    """
    Ajusta modelos de regresión para analizar cómo los diferentes volúmenes de ventas afectan AveragePrice.
    """
    print("Ajustando modelos de regresión para diferenciar volúmenes de ventas...")
    
    # Ajustar modelo de regresión
    X = Datos[['Total Volume', '4046', '4225', '4770']]
    y = Datos['AveragePrice']
    
    modelo_lineal = LinearRegression()
    modelo_lineal.fit(X, y)
    
    predicciones_lineales = modelo_lineal.predict(X)
    
    # Métricas
    mse_lineal = mean_squared_error(y, predicciones_lineales)
    r2_lineal = r2_score(y, predicciones_lineales)

    print(f"Modelo Lineal - MSE: {mse_lineal}, R²: {r2_lineal}")

def P5_9_InfluenciaVentas():
    """
    Analiza cómo varía AveragePrice en función del volumen total de ventas.
    """
    print("Analizando la influencia de las ventas totales en el precio promedio...")
    
    # Ajustar modelo de regresión
    X = Datos[['Total Volume']]
    y = Datos['AveragePrice']
    
    modelo = LinearRegression()
    modelo.fit(X, y)
    
    # Predicciones
    predicciones = modelo.predict(X)
    
    # Métricas
    mse = mean_squared_error(y, predicciones)
    r2 = r2_score(y, predicciones)
    
    print(f"MSE: {mse}, R²: {r2}")
