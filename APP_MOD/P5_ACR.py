from sklearn.ensemble import RandomForestRegressor 
from datetime import datetime
import time
from IPython.display import display, Markdown, HTML
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import statsmodels.api as sm 
from APPModels.APP_FUN import APP_Enunciados,chart
import APPModels.APP_FUN as app_fun  # Importa el módulo completo

from scipy.optimize import minimize
from itertools import combinations
from IPython.display import clear_output, display


# Modelos de Regresión en scikit-learn
# Regresión Lineal
from sklearn.linear_model import LinearRegression
# Regresión de Lasso (L1 regularization)
from sklearn.linear_model import Lasso
# Regresión Ridge (L2 regularization)
from sklearn.linear_model import Ridge
# Regresión ElasticNet (combina L1 y L2 regularization)
from sklearn.linear_model import ElasticNet
# Regresión de Árbol de Decisión
from sklearn.tree import DecisionTreeRegressor
# Regresión de Bosque Aleatorio (Random Forest)
from sklearn.ensemble import RandomForestRegressor
# Regresión de Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
# Regresión de Máquinas de Vectores de Soporte (SVR)
from sklearn.svm import SVR
# Regresión K-Vecinos (KNN)
from sklearn.neighbors import KNeighborsRegressor
# Regresión en XGBoost (si está instalado)
from xgboost import XGBRegressor
# Regresión en LightGBM (si está instalado)
import lightgbm as lgb

Datos = None
mDbg = None
def GEN_MatrizCorrelacion(pDf, pListaCampos=''):

    correlacion = pDf[pListaCampos].corr()
    

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlacion, annot=True, cmap='coolwarm')
    plt.title('Matriz de Correlación')
    plt.show()
    
    #print("Correlaciones significativas encontradas:")
    #print(correlacion)



    

def P5_1_MatrizCorrelacion(pListaCampos=''):

    APP_Enunciados.getEnunciado('5.1')
    #APP_Enunciados.getExplicacion('5.1')

    #display(Markdown(mDbg))
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
    APP_Enunciados.getEnunciado("5.2")
    APP_Enunciados.getExplicacion("5.2")

    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=Datos, x='Total Volume', y='AveragePrice')
    sns.regplot(data=Datos, x='Total Volume', y='AveragePrice', scatter=False, color='red')
    sns.regplot(data=Datos, x='Total Volume', y='AveragePrice', scatter=False, color='green',order=5)
    sns.regplot(data=Datos, x='Total Volume', y='AveragePrice', scatter=False, color='blue',order=6)
    sns.regplot(data=Datos, x='Total Volume', y='AveragePrice', scatter=False, color='black',order=7)
    plt.title('Análisis de Dispersión: AveragePrice vs Total Volume')


    # Escala logarítmica en el eje y
    plt.yscale('log')

    # Establecer los límites del eje x entre 1 y 1,000,000
    plt.xlim(1, 10000000)
    plt.show()
    
    print("El gráfico muestra la relación entre AveragePrice y Total Volume.")

def P5_2_AnalisisDispersiónN():
    APP_Enunciados.getEnunciado("5.2")
    APP_Enunciados.getExplicacion("5.2")


    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=Datos, x='AveragePrice', y='Total Volume')
    sns.regplot(data=Datos, x='AveragePrice', y='Total Volume', scatter=False, color='red')
    sns.regplot(data=Datos, x='AveragePrice', y='Total Volume', scatter=False, color='green',order=2)
    sns.regplot(data=Datos, x='AveragePrice', y='Total Volume', scatter=False, color='green',order=3)

    plt.title('Análisis de Dispersión:Total Volume vs AveragePrice  ')
    plt.show()
    
    print("El gráfico muestra la relación entre Total Volume y AveragePrice.")


def P5_3_PrediccionesMensuales( pModelo ='Media'):
    # Establece una altura de celda suficientemente grande
#interpolante, obtener los pesos.
    APP_Enunciados.getEnunciado('5.3')
    mDbg =f""""<span style='font-size:20px; color:blue; font-style:italic;'>
**promedio_precio = (Datos_trimestrales['AveragePrice'][i] + Datos_trimestrales['AveragePrice'][i + 1]) / 2**
</span>
    """

    SubDatos = app_fun.APP_DatosORG.copy()

    display(Markdown(mDbg))


    # Agrupación por trimestres y cálculo de promedios
    #Datos_trimestrales = SubDatos.groupby(pd.Grouper(key='CalFecha', freq='Q')).agg({'AveragePrice': 'mean', 'Total Volume': 'mean'}).reset_index()
    Datos_trimestrales = SubDatos.groupby(pd.Grouper(key='CalFecha', freq='Q')).agg({
            'AveragePrice': 'mean', 
            'Total Volume': 'mean'
        }).reset_index()
    
    Datos_trimestrales['Month'] = Datos_trimestrales['CalFecha'].dt.month
    # Predicciones
    predicciones = []
    fechas_prediccion = []
    for i in range(len(Datos_trimestrales) - 2):
        pModelo ='Media'
        if pModelo=='Media':
            promedio_precio = (Datos_trimestrales['AveragePrice'][i] + Datos_trimestrales['AveragePrice'][i + 1]) / 2
        elif pModelo =='MediaPodenrada':
            # Cálculo del promedio ponderado
            total_vol_0 = Datos_trimestrales['Total Volume'][i]
            total_vol_1 = Datos_trimestrales['Total Volume'][i + 1]
            avg_price_0 = Datos_trimestrales['AveragePrice'][i]
            avg_price_1 = Datos_trimestrales['AveragePrice'][i + 1]
            
            # Promedio ponderado de los precios utilizando el volumen total
            promedio_precio = ((avg_price_0 * total_vol_0) + (avg_price_1 * total_vol_1)) / (total_vol_0 + total_vol_1)

        elif pModelo=='MesAnt':
            promedio_precio =  Datos_trimestrales['AveragePrice'][i + 1]

        predicciones.append(promedio_precio)
        fechas_prediccion.append(Datos_trimestrales['CalFecha'][i + 2])  # Guardamos la fecha del tercer mes para el eje X
    
    # Comparar con los precios reales
    precios_reales = Datos_trimestrales['AveragePrice'][2:].reset_index(drop=True)
    predicciones_df = pd.DataFrame({
        'Fecha': fechas_prediccion,
        'Real': precios_reales,
        'Predicción': predicciones
    })

    # Agregar todas las fechas reales (incluyendo las que no tienen predicción)
    fechas_reales_completas = Datos_trimestrales['CalFecha'].reset_index(drop=True)  # Fechas reales completas
    precios_reales_completos = Datos_trimestrales['AveragePrice'].reset_index(drop=True)


    # Evaluación de precisión
    r2 = r2_score(predicciones_df['Real'], predicciones_df['Predicción'])
    rmse = np.sqrt(mean_squared_error(predicciones_df['Real'], predicciones_df['Predicción']))

    print(f"\nR²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Graficar resultados
    plt.figure(figsize=(10, 5))
    plt.title('Comparación de Precios Reales y Predicciones')
    plt.xlabel('Meses (a partir del tercer mes del trimestre)')

    # Graficar precios reales (todas las fechas)
    plt.plot(fechas_reales_completas, precios_reales_completos, label='Precios Reales', marker='o', color='blue')

    # Graficar predicciones
    plt.plot(predicciones_df['Fecha'], predicciones_df['Predicción'], label='Predicciones', marker='o', color='red')

    # (Opcional) Graficar resultados
    #plt.figure(figsize=(10, 5))
    #plt.plot(predicciones_df['Fecha'],predicciones_df['Real'], label='Precios Reales', marker='o')
    #plt.plot(predicciones_df['Fecha'],predicciones_df['Predicción'], label='Predicciones', marker='o')
    #plt.xticks(predicciones_df['Fecha'], predicciones_df['Fecha'].dt.strftime('%Y-%m'), rotation=45)  # Formato año-mes
   # Formato de fecha en el eje X
    plt.xticks(fechas_reales_completas, fechas_reales_completas.dt.strftime('%Y-%m'), rotation=45)
      
    plt.ylabel('Average Price')
    plt.legend()
    plt.grid()
    #plt.tight_layout()
    plt.show()


    #GEN_MatrizCorrelacion(predicciones_df,['Fecha','Real','Predicción'])

    #Elimina la primera fila que tiene las predicciones en nan

    predicciones_df = predicciones_df.drop([0, 1]).reset_index(drop=True)
    if 1==2: #no quiero que lo imprima por ahora, pero que sea visible el codigo
        GEN_MatrizCorrelacion(predicciones_df,['Fecha','Real','Predicción'])
    




def P5_4_PrediccionesTrimestrales():
    APP_Enunciados.getEnunciado("5.4")
    APP_Enunciados.getExplicacion("5.4")
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

        # Diferencias en porcentaje
        dif_pred_pct = round((prediccion_lineal[-1] - prediccion_polinomica[-1]) / prediccion_lineal[-1] * 100, 2) 
        dif_r2_pct = round((r2_lineal - r2_polinomico) / abs(r2_lineal) * 100, 2)  if r2_lineal != 0 else 0
        dif_rmse_pct = round((rmse_polinomico - rmse_lineal) / rmse_lineal * 100, 2)  if rmse_lineal != 0 else 0
        """
        # Interpretación de las diferencias
        interpr_dif_pred = '.' if abs(dif_pred_pct) < 5 else 'M' if abs(dif_pred_pct) <= 20 else 'S'
        interpr_dif_r2 = '' if dif_r2_pct < 0 else 'KO'
        interpr_dif_rmse = '' if dif_rmse_pct < 0 else 'KO'
        """

        # Interpretación de las diferencias con prefijo '+' o '-'
        interpr_dif_pred = (f"+{'.' if abs(dif_pred_pct) < 5 else 'M' if abs(dif_pred_pct) <= 20 else 'S'}" 
                            if dif_pred_pct >= 0 else 
                            f"-{'.' if abs(dif_pred_pct) < 5 else 'M' if abs(dif_pred_pct) <= 20 else 'S'}")
        
        interpr_dif_r2 = ('' if dif_r2_pct < 0 else f"+KO" if dif_r2_pct >= 0 else f"-KO")
        
        interpr_dif_rmse = ('' if dif_rmse_pct < 0 else f"+KO" if dif_rmse_pct >= 0 else f"-KO")


        # Añadir resultados al DataFrame
        resultados_df = pd.concat([resultados_df, pd.DataFrame({
            'Columna': [columna],
            'PrediccionLineal': [prediccion_lineal[-1]],
            'PrediccionPolinómica': [prediccion_polinomica[-1]],
            'R² Lineal': [r2_lineal],
            'R² Polinómica': [r2_polinomico],            
            'RMSE Lineal': [rmse_lineal],
            'RMSE Polinómica': [rmse_polinomico],
            'Dif Pred_%': [dif_pred_pct],
            'Interpr Dif Pred_%': [interpr_dif_pred],
            'Dif R2_%': [dif_r2_pct],
            'Interpr Dif R2_%': [interpr_dif_r2],
            'Dif RMSE_%': [dif_rmse_pct],
            'Interpr Dif RMSE_%': [interpr_dif_rmse]            
        })], ignore_index=True)
    # Mostrar el DataFrame de resultados
    display(resultados_df)
        # Crear variable de texto con la interpretación en Markdown
    interpretacion_md = """

### Interpretación de las diferencias en porcentaje:

   - **Interpr_Dif_Pred_%**: Indicador del cambio en la predicción.
        - `+.` : Cambio pequeño positivo en la predicción (< 5%)
        - `-.` : Cambio pequeño negativo en la predicción (< 5%)
        - `+M` : Cambio moderado positivo en la predicción (5% - 20%)
        - `-M` : Cambio moderado negativo en la predicción (5% - 20%)
        - `+S` : Cambio significativo positivo en la predicción (> 20%)
        - `-S` : Cambio significativo negativo en la predicción (> 20%)
        
   - **Interpr_Dif_R2_%**: Indica si el ajuste del modelo mejora o empeora.
        - ''  : Mejora el ajuste en el modelo polinómico
        - `+KO`: Empeora el ajuste en el modelo polinómico

   - **Interpr_Dif_RMSE_%**: Indica si hay un cambio notable en el error.
        - ''  : Mejora (disminuye el error) en el modelo polinómico
        - `+KO`: Empeora (aumenta el error) en el modelo polinómico
    """
    display(Markdown(interpretacion_md))
    
def P5_5_PrediccionesAnuales(anios_previos=1):
    APP_Enunciados.getEnunciado("5.5")
    APP_Enunciados.getExplicacion("5.5")

    mDbg = f"""- **parametros**:  
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
    # Exportar el DataFrame a un fichero Excel
    df_resultados.to_excel('P5_5_PrediccionesAnuales.xlsx', index=False)
    print(f"Resultados exportados a: P5_5_PrediccionesAnuales")

def P5_6_Modelos_Regresión_Múltiple():
    APP_Enunciados.getEnunciado("5.6")
    APP_Enunciados.getExplicacion("5.6")

    global Datos  # Asegurarse de que los datos están accesibles
    
    # Filtrar columnas relevantes
    features = ['Total Volume', '4046', '4225', '4770','Large Bags', 'Total Bags']
    target = 'AveragePrice'
    

    # Definir variables independientes (X) y dependientes (y)
    X = Datos[features]
    y = Datos[target]

    # Ajustar el modelo de regresión lineal
    modelo_regresion = LinearRegression()
    modelo_regresion.fit(X, y)

    # Realizar predicciones
    predicciones = modelo_regresion.predict(X)

    # Calcular métricas de rendimiento
    r2 = r2_score(y, predicciones)
    rmse = mean_squared_error(y, predicciones, squared=False)

    
    # Mostrar los resultados
    print("Resultados del Modelo de Regresión Múltiple:")
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Coeficientes del modelo
    coeficientes = pd.DataFrame(modelo_regresion.coef_, features, columns=['Coeficiente'])
    print("\nCoeficientes del Modelo:")
    print(coeficientes)

    # Implicaciones de los resultados
    print("\nImplicaciones:")
    if r2 < 0:
        print("El modelo no es adecuado, ya que el R² es negativo.")
    elif r2 < 0.5:
        print("El modelo tiene un ajuste bajo, lo que sugiere que las variables seleccionadas no explican bien la variabilidad de AveragePrice.")
    elif r2 < 0.8:
        print("El modelo tiene un ajuste moderado. Hay margen para mejorar la precisión mediante la selección de características o modelos alternativos.")
    else:
        print("El modelo tiene un buen ajuste y las variables seleccionadas explican bien la variabilidad de AveragePrice.")


def P5_7_CoefficientsRegresionMultiple():
    APP_Enunciados.getEnunciado("5.7")
    APP_Enunciados.getExplicacion("5.7")

    
    global Datos  # Asegurarse de que los datos están accesibles
    
    # Filtrar columnas relevantes
    features = ['Total Volume', '4046', '4225', '4770', 'Large Bags','Total Bags']
    target = 'AveragePrice'
    
    # Comprobar si las columnas existen en los datos
    if not all(col in Datos.columns for col in features + [target]):
        print("Faltan algunas columnas necesarias en los datos.")
        return

    # Definir variables independientes (X) y dependientes (y)
    X = Datos[features]
    y = Datos[target]

    # Ajustar el modelo de regresión lineal
    modelo_regresion = LinearRegression()
    modelo_regresion.fit(X, y)

    # Extraer los coeficientes del modelo
    coeficientes = modelo_regresion.coef_

    # Crear un DataFrame para visualizar los coeficientes
    df_coeficientes = pd.DataFrame(coeficientes, index=features, columns=['Coeficiente'])
    df_coeficientes['Interpretación'] = df_coeficientes['Coeficiente'].apply(lambda x: "Aumenta" if x > 0 else "Disminuye")


    # Mostrar los coeficientes y su interpretación
    print("Análisis de Coeficientes de Regresión Múltiple:")
    print(df_coeficientes)

    # Comentar sobre las variables más significativas
    variables_significativas = df_coeficientes[df_coeficientes['Coeficiente'].abs() > 0.1]  # Definir un umbral para significancia
    print("\nVariables más significativas:")
    print(variables_significativas)

    print("\nComentarios sobre las variables:")
    for var, row in variables_significativas.iterrows():
        impacto = row['Coeficiente']
        if impacto > 0:
            print(f"- Un aumento en {var} de una unidad incrementa el precio promedio en {impacto:.6f}.")
        else:
            print(f"- Un aumento en {var} de una unidad disminuye el precio promedio en {abs(impacto):.6f}.")



def P5_8_Regresion_VolumenVentas():
    APP_Enunciados.getEnunciado("5.8")
    APP_Enunciados.getExplicacion("5.8")


    global Datos
    
    # Verificamos si las columnas necesarias están en el DataFrame
    required_columns = {'AveragePrice', 'Total Volume', '4046', '4225', '4770'}
    if not required_columns.issubset(Datos.columns):
        raise ValueError("Los datos no contienen todas las columnas necesarias: " + str(required_columns))
    
    # Seleccionamos las columnas de interés
    X = Datos[['Total Volume', '4046', '4225', '4770']]
    y = Datos['AveragePrice']
    
    # Dividimos los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Modelo de Regresión Lineal
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)
    
    # Evaluación del modelo lineal
    mse_linear = mean_squared_error(y_test, y_pred_linear)
    r2_linear = r2_score(y_test, y_pred_linear)
    
    # Modelo de Regresión Polinómica (grado 2)
    poly_features = PolynomialFeatures(degree=2)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    y_pred_poly = poly_model.predict(X_test_poly)
    
    # Evaluación del modelo polinómico
    mse_poly = mean_squared_error(y_test, y_pred_poly)
    r2_poly = r2_score(y_test, y_pred_poly)
    
    # Imprimimos los resultados de las métricas para comparar
    print("Resultados de la Regresión Lineal:")
    print(f"Error Cuadrático Medio (MSE): {mse_linear:.4f}")
    print(f"Coeficiente de Determinación (R^2): {r2_linear:.4f}")
    print("\nResultados de la Regresión Polinómica (Grado 2):")
    print(f"Error Cuadrático Medio (MSE): {mse_poly:.4f}")
    print(f"Coeficiente de Determinación (R^2): {r2_poly:.4f}")
    
    # Conclusiones
    if r2_poly > r2_linear:
        print("\nEl modelo de regresión polinómica proporciona un mejor ajuste a los datos en comparación con el modelo lineal.")
    else:
        print("\nEl modelo de regresión lineal proporciona un ajuste comparable o mejor que el modelo polinómico para estos datos.")

def P5_8_VolumenVentas():
    APP_Enunciados.getEnunciado("5.8")
    APP_Enunciados.getExplicacion("5.8")

    
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

def P5_9_AnalisisInfluenciaVentas():
    APP_Enunciados.getEnunciado("5.9")
    APP_Enunciados.getExplicacion("5.9")

    
    global Datos  # Asegurarse de que los datos están accesibles

    # Seleccionar las variables de interés
    X = Datos[['Total Volume', 'Total Bags']]  # Variables independientes
    y = Datos['AveragePrice']  # Variable dependiente

    # Agregar un término constante para el modelo lineal
    X = sm.add_constant(X)  # Agregar término independiente para statsmodels

    # Ajuste del modelo de regresión lineal
    modelo_lineal = sm.OLS(y, X).fit()

    # Predicciones y evaluación del modelo lineal
    predicciones_lineales = modelo_lineal.predict(X)
    r2_lineal = modelo_lineal.rsquared
    rmse_lineal = np.sqrt(mean_squared_error(y, predicciones_lineales))

    # Imprimir resultados del modelo lineal
    print("Resultados del Modelo de Regresión Lineal:")
    print(modelo_lineal.summary())
    print(f"R² Lineal: {r2_lineal:.6f}")
    print(f"RMSE Lineal: {rmse_lineal:.6f}\n")

    # Ajuste del modelo de regresión polinómica (grados 2)
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)  # Transformar las variables independientes
    modelo_polinomico = sm.OLS(y, X_poly).fit()

    # Predicciones y evaluación del modelo polinómico
    predicciones_polinomicas = modelo_polinomico.predict(X_poly)
    r2_polinomico = modelo_polinomico.rsquared
    rmse_polinomico = np.sqrt(mean_squared_error(y, predicciones_polinomicas))

    # Imprimir resultados del modelo polinómico
    print("Resultados del Modelo de Regresión Polinómica:")
    print(modelo_polinomico.summary())
    print(f"R² Polinómico: {r2_polinomico:.6f}")
    print(f"RMSE Polinómico: {rmse_polinomico:.6f}")

def P5_10_RegresionPrecioPromedioPorTipo():
    APP_Enunciados.getEnunciado("5.10")
    APP_Enunciados.getExplicacion("5.10")

    
    global Datos  # Asegurarse de que los datos están accesibles

    # Seleccionar las variables de interés
    X = Datos[['4046', '4225', '4770', 'Total Volume']]  # Variables independientes
    y = Datos['AveragePrice']  # Variable dependiente

    # Agregar un término constante para el modelo lineal
    X = sm.add_constant(X)  # Agregar término independiente para statsmodels

    # Ajuste del modelo de regresión lineal
    modelo_lineal = sm.OLS(y, X).fit()

    # Predicciones y evaluación del modelo lineal
    predicciones_lineales = modelo_lineal.predict(X)
    r2_lineal = modelo_lineal.rsquared
    rmse_lineal = np.sqrt(mean_squared_error(y, predicciones_lineales))

    # Imprimir resultados del modelo lineal
    print("Resultados del Modelo de Regresión Lineal:")
    print(modelo_lineal.summary())
    print(f"R² Lineal: {r2_lineal:.6f}")
    print(f"RMSE Lineal: {rmse_lineal:.6f}\n")

    # Ajuste del modelo de regresión polinómica (grados 2)
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)  # Transformar las variables independientes
    modelo_polinomico = sm.OLS(y, X_poly).fit()

    # Predicciones y evaluación del modelo polinómico
    predicciones_polinomicas = modelo_polinomico.predict(X_poly)
    r2_polinomico = modelo_polinomico.rsquared
    rmse_polinomico = np.sqrt(mean_squared_error(y, predicciones_polinomicas))

    # Imprimir resultados del modelo polinómico
    print("Resultados del Modelo de Regresión Polinómica:")
    print(modelo_polinomico.summary())
    print(f"R² Polinómico: {r2_polinomico:.6f}")
    print(f"RMSE Polinómico: {rmse_polinomico:.6f}")

    # Comparación de modelos
    if r2_lineal > r2_polinomico:
        print("El modelo de regresión lineal ofrece mejores predicciones según el R².")
    else:
        print("El modelo de regresión polinómica ofrece mejores predicciones según el R².")

    if rmse_lineal < rmse_polinomico:
        print("El modelo de regresión lineal tiene un RMSE más bajo, lo que indica mejor precisión.")
    else:
        print("El modelo de regresión polinómica tiene un RMSE más bajo, lo que indica mejor precisión.")
