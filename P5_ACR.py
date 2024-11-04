from IPython.display import display, Markdown, HTML
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


Datos =''    



def P5_1_MatrizCorrelacion(pListaCampos=''):
    """
    Genera y visualiza la matriz de correlación entre las variables numéricas del DataFrame.
    """
    print("Generando matriz de correlación...")
    
    #correlacion = Datos[['AveragePrice', 'Total Volume', '4046', '4225', '4770', 'Total Bags']].corr()
    #correlacion = Datos[['AveragePrice', 'Total Volume', '4046', '4225', '4770', 'Total Bags','Small Bags','Large Bags','XLarge Bags','year']].corr()
    correlacion = Datos[pListaCampos].corr()
    

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlacion, annot=True, cmap='coolwarm')
    plt.title('Matriz de Correlación')
    plt.show()
    
    print("Correlaciones significativas encontradas:")
    print(correlacion)

def P5_2_AnalisisDispersión():
    """
2. **Análisis de Dispersión entre Variables Clave:** 
   - **Uso de Datos:** Selecciona variables numéricas de interés como `AveragePrice` y `Total Volume`.
   - **Esperado:** 
     - Importa las librerías necesarias: `import seaborn as sns` y `import matplotlib.pyplot as plt`.
     - Crea un gráfico de dispersión con `sns.scatterplot()` para visualizar la relación entre `AveragePrice` y `Total Volume`.
     - Añade una línea de regresión utilizando `sns.regplot()` para ilustrar las tendencias.
     - Compara el ajuste de una regresión lineal frente a una polinómica.
    """
    mDbg =P5_2_AnalisisDispersión.__doc__


    display(Markdown(mDbg))

    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=Datos, x='Total Volume', y='AveragePrice')
    sns.regplot(data=Datos, x='Total Volume', y='AveragePrice', scatter=False, color='red')
    sns.regplot(data=Datos, x='Total Volume', y='AveragePrice', scatter=False, color='green',order=2)
    sns.regplot(data=Datos, x='Total Volume', y='AveragePrice', scatter=False, color='green',order=3)
    plt.title('Análisis de Dispersión: AveragePrice vs Total Volume')
    plt.show()
    
    print("El gráfico muestra la relación entre AveragePrice y Total Volume.")

def P5_2_AnalisisDispersiónN():
    """
2. **Análisis de Dispersión entre Variables Clave:** 
   - **Uso de Datos:** Selecciona variables numéricas de interés como `AveragePrice` y `Total Volume`.
   - **Esperado:** 
     - Importa las librerías necesarias: `import seaborn as sns` y `import matplotlib.pyplot as plt`.
     - Crea un gráfico de dispersión con `sns.scatterplot()` para visualizar la relación entre `AveragePrice` y `Total Volume`.
     - Añade una línea de regresión utilizando `sns.regplot()` para ilustrar las tendencias.
     - Compara el ajuste de una regresión lineal frente a una polinómica.
    """
    mDbg =P5_2_AnalisisDispersión.__doc__


    display(Markdown(mDbg))

    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=Datos, x='AveragePrice', y='Total Volume')
    sns.regplot(data=Datos, x='AveragePrice', y='Total Volume', scatter=False, color='red')
    sns.regplot(data=Datos, x='AveragePrice', y='Total Volume', scatter=False, color='green',order=2)
    sns.regplot(data=Datos, x='AveragePrice', y='Total Volume', scatter=False, color='green',order=3)

    plt.title('Análisis de Dispersión:Total Volume vs AveragePrice  ')
    plt.show()
    
    print("El gráfico muestra la relación entre AveragePrice y Total Volume.")


def P5_3_PrediccionesMensuales():

    """
3. **Predicciones Mensuales Usando Datos Trimestrales:**
   - **Uso de Datos:** Agrupa datos por trimestres y segmenta en meses utilizando `Date`, `AveragePrice`, y `Total Volume`.
   - **Esperado:** 
     - Convierte la columna `Date` a tipo datetime si es necesario.
     - Agrupa los datos por trimestre y calcula el promedio de `AveragePrice` y `Total Volume`.
     - Utiliza los datos de los primeros 2 meses de un trimestre para predecir el precio del tercer mes.
     - Compara los resultados de las predicciones con los precios reales.
     - Evalúa la precisión de tus predicciones utilizando métricas como R² y RMSE.
         
    """
    mDbg =P5_3_PrediccionesMensuales.__doc__


    display(Markdown(mDbg))

    
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
    # Evaluación de precisión
    r2 = r2_score(predicciones_df['Real'], predicciones_df['Predicción'])
    rmse = np.sqrt(mean_squared_error(predicciones_df['Real'], predicciones_df['Predicción']))

    print(f"\nR²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # (Opcional) Graficar resultados
    plt.figure(figsize=(10, 5))
    plt.plot(predicciones_df['Real'], label='Precios Reales', marker='o')
    plt.plot(predicciones_df['Predicción'], label='Predicciones', marker='o')
    plt.title('Comparación de Precios Reales y Predicciones')
    plt.xlabel('Meses (a partir del tercer mes del trimestre)')
    plt.ylabel('Average Price')
    plt.legend()
    plt.grid()
    plt.show()


def P5_4_PrediccionesTrimestrales():
    """
4. **Predicciones Trimestrales:**
   - **Uso de Datos:** Agrupa los datos en trimestres usando solo variables numéricas.
   - **Esperado:** 
     - Agrupa los datos por trimestres usando `pd.Grouper()` con `freq='Q'` para obtener promedios.
     - Usa los datos de 1 o 2 trimestres anteriores para predecir el siguiente trimestre ajustando modelos de regresión lineal y polinómica.
     - Compara los resultados de las predicciones con los precios reales.
     - Evalúa la precisión de tus predicciones utilizando métricas como R² y RMSE    
    """
    mDbg =P5_4_PrediccionesTrimestrales.__doc__


    display(Markdown(mDbg))
    global Datos
    # Filtrar solo las columnas numéricas para agrupar
    datos_trimestrales = Datos.groupby(pd.Grouper(key='CalFecha', freq='Q')).mean(numeric_only=True).reset_index()
    # Lista de resultados para cada columna
    
    # Crear un DataFrame vacío para almacenar los resultados
    resultados_df = pd.DataFrame(columns=[
        'Columna', 'PrediccionLineal', 'PrediccionPolinómica', 
        'R² Lineal', 'R² Polinómica', 'RMSE Lineal', 'RMSE Polinómica'
    ])
    # Iterar sobre cada columna numérica (omitimos 'CalFecha' ya que es la columna de fechas)
    for columna in datos_trimestrales.columns:
        if columna == 'CalFecha':
            continue
        
        # Preparar datos para regresión
        X = np.arange(len(datos_trimestrales)).reshape(-1, 1)  # Índice de tiempo para la regresión
        y = datos_trimestrales[columna].values

        # Modelo de regresión lineal
        modelo_lineal = LinearRegression()
        modelo_lineal.fit(X[:-1], y[:-1])  # Entrenar con todos menos el último trimestre
        prediccion_lineal = modelo_lineal.predict(X)

        # Modelo de regresión polinómica de grado 2
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X[:-1])
        modelo_polinomico = LinearRegression()
        modelo_polinomico.fit(X_poly, y[:-1])
        prediccion_polinomica = modelo_polinomico.predict(poly.transform(X))

        # Calcular métricas de precisión
        r2_lineal = r2_score(y, prediccion_lineal)
        rmse_lineal = mean_squared_error(y, prediccion_lineal, squared=False)
        r2_polinomico = r2_score(y, prediccion_polinomica)
        rmse_polinomico = mean_squared_error(y, prediccion_polinomica, squared=False)


        # Añadir resultados al DataFrame
        resultados_df = pd.concat([resultados_df, pd.DataFrame({
            'Columna': [columna],
            'PrediccionLineal': [prediccion_lineal[-1]],
            'PrediccionPolinómica': [prediccion_polinomica[-1]],
            'R² Lineal': [r2_lineal],
            'R² Polinómica': [r2_polinomico],            
            'RMSE Lineal': [rmse_lineal],
            'RMSE Polinómica': [rmse_polinomico]
        })], ignore_index=True)
    # Mostrar el DataFrame de resultados
    display(resultados_df)
    
def P5_5_PrediccionesAnuales(anios_previos=1):
    """
5. **Predicciones Anuales:**
   - **Uso de Datos:** Agrupa los datos en años, utilizando únicamente columnas numéricas.
   - **Esperado:** 
     - Agrupa los datos por año utilizando `pd.Grouper()` con `freq='Y'`.
     - Usa los datos de 1 o 2 años anteriores para predecir el siguiente año ajustando modelos de regresión lineal y polinómica.
     - Evalúa la precisión de tus predicciones utilizando métricas como R² y RMSE.


    """
    mDbg =P5_5_PrediccionesAnuales.__doc__


    mDbg += f"""- **parametros**:  
         - **anios_previos:**`{anios_previos}` 
    """

    display(Markdown(mDbg))

    
    """
    Parámetros:
    - `anios_previos`: int, número de años previos para entrenamiento del modelo (1 o 2).
    
    Retorna:
    - DataFrame con columnas: ['Columna', 'PrediccionLineal', 'PrediccionPolinómica', 
                               'R² Lineal', 'R² Polinómica', 'RMSE Lineal', 'RMSE Polinómica'].
    """
    global Datos  # Asegurarse de que los datos están accesibles
    
    # Agrupación de datos anuales con medias
    datos_anuales = Datos.groupby(pd.Grouper(key='CalFecha', freq='Y')).mean(numeric_only=True).reset_index()

    resultados = []

    # Iterar sobre cada columna numérica (omitimos 'CalFecha' ya que es la columna de fechas)
    for columna in datos_anuales.columns:
        if columna == 'CalFecha':
            continue
        
        # Preparar datos para regresión
        X = np.arange(len(datos_anuales)).reshape(-1, 1)  # Índice de tiempo para la regresión
        y = datos_anuales[columna].values

        # Limitar los datos según el parámetro `anios_previos`
        if len(X) <= anios_previos:
            continue  # Evitar predicciones si no hay suficientes datos
        
        X_train = X[:-1] if anios_previos == 1 else X[:-2]
        y_train = y[:-1] if anios_previos == 1 else y[:-2]

        # Modelo de regresión lineal
        modelo_lineal = LinearRegression()
        modelo_lineal.fit(X_train, y_train)
        prediccion_lineal = modelo_lineal.predict(X)

        # Modelo de regresión polinómica de grado 2
        poly = PolynomialFeatures(degree=2)
        X_poly_train = poly.fit_transform(X_train)
        modelo_polinomico = LinearRegression()
        modelo_polinomico.fit(X_poly_train, y_train)
        prediccion_polinomica = modelo_polinomico.predict(poly.transform(X))

        # Calcular métricas de precisión
        r2_lineal = r2_score(y, prediccion_lineal)
        rmse_lineal = mean_squared_error(y, prediccion_lineal, squared=False)
        r2_polinomico = r2_score(y, prediccion_polinomica)
        rmse_polinomico = mean_squared_error(y, prediccion_polinomica, squared=False)

        # Guardar los resultados en una lista de diccionarios
        resultados.append({
            'Columna': columna,
            'PrediccionLineal': prediccion_lineal[-1],
            'PrediccionPolinómica': prediccion_polinomica[-1],
            'R² Lineal': r2_lineal,
            'R² Polinómica': r2_polinomico,
            'RMSE Lineal': rmse_lineal,
            'RMSE Polinómica': rmse_polinomico
        })

    print ('sss')

    # Convertir los resultados a DataFrame para visualización y retorno
    df_resultados = pd.DataFrame(resultados)
    display(df_resultados)


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
