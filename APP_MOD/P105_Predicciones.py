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

gPre = None
Datos_mensuales =None
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

def P105_3_PrediccionesMensuales(pModelo='Media'):
    global Datos_mensuales
    # Obtener enunciado
    APP_Enunciados.getEnunciado('5.3')
    
    mDbg =f"""<span style='font-size:20px; color:blue; font-style:italic;'>
    **promedio_precio = (Datos_mensuales['AveragePrice'][i] + Datos_mensuales['AveragePrice'][i + 1]) / 2**
    </span>"""
    
    SubDatos = app_fun.APP_DatosORG.copy()
    display(Markdown(mDbg))

    # Agrupación por meses y cálculo de promedios
    Datos_mensuales = SubDatos.groupby(pd.Grouper(key='CalFecha', freq='M')).agg({
        'AveragePrice': 'mean', 
        'AveragePrice_PREV': 'mean',         
        'Total Volume': 'mean'
    }).reset_index()
    
    Datos_mensuales['Month'] = Datos_mensuales['CalFecha'].dt.month

    # Predicciones
    predicciones = []
    fechas_prediccion = []
    for i in range(len(Datos_mensuales) - 2):
        if pModelo == 'Media':
            promedio_precio = (Datos_mensuales['AveragePrice'][i] + Datos_mensuales['AveragePrice'][i + 1]) / 2
        elif pModelo == 'MediaPonderada':
            # Cálculo del promedio ponderado
            total_vol_0 = Datos_mensuales['Total Volume'][i]
            total_vol_1 = Datos_mensuales['Total Volume'][i + 1]
            avg_price_0 = Datos_mensuales['AveragePrice'][i]
            avg_price_1 = Datos_mensuales['AveragePrice'][i + 1]
            
            promedio_precio = ((avg_price_0 * total_vol_0) + (avg_price_1 * total_vol_1)) / (total_vol_0 + total_vol_1)

        elif pModelo == 'MesAnt':
            promedio_precio = Datos_mensuales['AveragePrice'][i + 1]

        predicciones.append(promedio_precio)
        fechas_prediccion.append(Datos_mensuales['CalFecha'][i + 2])  # Guardamos la fecha del tercer mes para el eje X
    
    # Comparar con los precios reales
    precios_reales = Datos_mensuales['AveragePrice'][2:].reset_index(drop=True)
    precios_reales_PREV = Datos_mensuales['AveragePrice_PREV'][2:].reset_index(drop=True)
    
    predicciones_df = pd.DataFrame({
        'Fecha': fechas_prediccion,
        'Real': precios_reales,
        'Real_PREV': precios_reales_PREV,        
        'Predicción': predicciones
    })

    # Agregar todas las fechas reales (incluyendo las que no tienen predicción)
    fechas_reales_completas = Datos_mensuales['CalFecha'].reset_index(drop=True)
    precios_reales_completos = Datos_mensuales['AveragePrice'].reset_index(drop=True)
    precios_reales_PREV_completos = Datos_mensuales['AveragePrice_PREV'].reset_index(drop=True)

    

    # Evaluación de precisión
    r2 = r2_score(predicciones_df['Real'], predicciones_df['Predicción'])
    rmse = np.sqrt(mean_squared_error(predicciones_df['Real'], predicciones_df['Predicción']))

    # Evaluación de precisión
    r2_PREV = r2_score(predicciones_df['Real'], predicciones_df['Real_PREV'])
    rmse_PREV = np.sqrt(mean_squared_error(predicciones_df['Real'], predicciones_df['Real_PREV']))

    r2_real = r2_score(Datos_mensuales['AveragePrice'], Datos_mensuales['AveragePrice_PREV'])
    r2_realT = r2_score(SubDatos['AveragePrice'], SubDatos['AveragePrice_PREV'])


    print(f"\nR²: {r2:.6f}")
    print(f"RMSE: {rmse:.6f}")

    print(f"\nR²_PREV: {r2_PREV:.6f}")
    print(f"RMSE_PREV: {rmse_PREV:.6f}")
    print(f"r2_real datos agrupadas por mes : {r2_real:.6f}")
    print(f"r2_real datos DESGLOSADOS: {r2_realT:.6f}")

    # Graficar resultados
    plt.figure(figsize=(10, 5))
    plt.title('Comparación de Precios Reales y Predicciones')
    plt.xlabel('Meses')

    # Graficar precios reales (todas las fechas)
    plt.plot(fechas_reales_completas, precios_reales_completos, label='Precios Reales', marker='o', color='blue')

    # Graficar precios reales (todas las fechas)
    plt.plot(fechas_reales_completas, precios_reales_PREV_completos, label='Precios Reales PREV', marker='o', color='green')

    # Graficar predicciones
    plt.plot(predicciones_df['Fecha'], predicciones_df['Predicción'], label='Predicciones', marker='o', color='red')

    # Formato de fecha en el eje X
    plt.xticks(fechas_reales_completas, fechas_reales_completas.dt.strftime('%Y-%m'), rotation=45)
      
    plt.ylabel('Average Price')
    plt.legend()
    plt.grid()
    plt.show()

    # Elimina las primeras filas sin predicciones
    predicciones_df = predicciones_df.drop([0, 1]).reset_index(drop=True)
    global gPre
    gPre = predicciones_df
    if 1 == 1:
        GEN_MatrizCorrelacion(predicciones_df, ['Fecha', 'Real', 'Predicción'])

def GEN_MatrizCorrelacion(pDf, pListaCampos=''):

    correlacion = pDf[pListaCampos].corr(method='pearson')
    

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlacion, annot=True, cmap='coolwarm')
    plt.title('Matriz de Correlación')
    plt.show()
    
    print("Correlaciones significativas encontradas:")
    print(correlacion)

def P100_1_Modelo_TRAINING_Mod(pNameModelo ='LinearRegression',pFechaReal='2018-10-01' ):

    vModelo =LinearRegression()
    if pNameModelo == 'LinearRegression':
        vModelo =LinearRegression()
    elif pNameModelo == 'RandomForestRegressor':
        
        vModelo =RandomForestRegressor(n_estimators=100, random_state=42)

    print(vModelo.__class__.__name__)
    print(pFechaReal)
    

    RandomForestRegressor(n_estimators=100, random_state=42)
    # Mostrar el enunciado
    SubDatos = app_fun.APP_DatosORG.copy()
    print(len(SubDatos))
    SubDatos = SubDatos[SubDatos['CalFecha'] <= pFechaReal]
    print(len(SubDatos))

        # Definir las columnas independientes y dependientes
    campo_independiente = "AveragePrice"
    #campos_dependientes = ["Total Volume", "4046", "4225", "4770", "Total Bags", "Small Bags", "Large Bags", "XLarge Bags", "Cal_AAAAMM", "Cal_AAAA", "Cal_MM","Cal_SS","Cal_DDD","Cal_AAAADDD","CalNOR_MM_TotalVolume"]
    
    #campos_dependientes = ["CalNOR_MM_TotalVolume", "Total Volume",  "Cal_AAAA", "Cal_MM",'Cal_DDD']
    campos_dependientes = ["CalNOR_MM_TotalVolume",  "Cal_AAAA", "Cal_MM",'Cal_DDD']
    #campos_dependientes = ["Total Volume","Cal_MM"]

        # Separar los datos en características (X) y objetivo (y)
    X = SubDatos[campos_dependientes]
    y = SubDatos[campo_independiente]
    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f'X_train LEN: {len(X_train)}')

        # Entrenar el modelo
    vModelo.fit(X_train, y_train)
        
    # Evaluación en el conjunto de real
    X_real = SubDatos[campos_dependientes]
    y_real = SubDatos[campo_independiente]
    y_real_pred = vModelo.predict(X_real)
    # Evaluación de la predicción en el conjunto real
    mse_real = mean_squared_error(y_real, y_real_pred)
    rmse_real = np.sqrt(mse_real)
    r2_real = r2_score(y_real, y_real_pred)
    # Imprimir las métricas de evaluación para el conjunto real
    print(f"MSE (Real): {mse_real:.6f}")
    print(f"RMSE (Real): {rmse_real:.6f}")
    print(f"R² (Real): {r2_real:.6f}")

    # Variables y modelo previamente definido
    nuevo_campo_prediccion = campo_independiente + "_PREV"

    # Generar las predicciones para todo el conjunto 'SubDatos'
    SubDatos[nuevo_campo_prediccion] = vModelo.predict(SubDatos[campos_dependientes])

    # Calcular las métricas de evaluación en el conjunto real usando SubDatos
    mse_real = mean_squared_error(SubDatos[campo_independiente], SubDatos[nuevo_campo_prediccion])
    rmse_real = np.sqrt(mse_real)
    r2_real = r2_score(SubDatos[campo_independiente], SubDatos[nuevo_campo_prediccion])

    # Imprimir las métricas de evaluación
    print(f"MSE (Real): {mse_real:.6f}")
    print(f"RMSE (Real): {rmse_real:.6f}")
    print(f"R² (Real): {r2_real:.6f}")

    # Evaluación en el conjunto de entrenamiento
    y_train_pred = vModelo.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(y_train, y_train_pred)

    # Evaluación en el conjunto de prueba
    y_test_pred = vModelo.predict(X_test)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_test_pred)
    print('Test Evaluación')
    print('**************')
    print(f'mse_train:{mse_train}')
    print(f'rmse_train:{rmse_train}')
    print(f'r2_train:{r2_train}')

    # Evaluación en el conjunto de prueba
    print(f'mse_test:{mse_test}')
    print(f'rmse_test:{rmse_test}' )
    print(f'r2_test:{r2_test}')
    print('**************')

    vCoef = ''
    vCoef = P99_1_Modelo_TRAINING_FIT_Coef(vModelo,campo_independiente,campos_dependientes)
    print(vCoef)
    # Aplicar el modelo a los datos org
    #app_fun.APP_DatosORG[campo_independiente + '_PREV'] = vModelo.predict(SubDatos[campos_dependientes])
    app_fun.APP_DatosORG[campo_independiente + '_PREV'] = None
    app_fun.APP_DatosORG[campo_independiente + '_PREV'] = vModelo.predict(app_fun.APP_DatosORG[campos_dependientes])

def P99_1_Modelo_TRAINING_FIT_Coef(pModelo, campo_independiente, campos_dependientes):
        # 6. Mostrar los pesos de las características
    vRes =''
    # Verificar si el modelo tiene el atributo 'coef_' (tipico de modelos lineales)
    if hasattr(pModelo, 'coef_'):

        coeficientes_features  = pModelo.coef_
        importancia_df = pd.DataFrame({
            'Característica': campos_dependientes,
            'Coeficiente': coeficientes_features 
        }).sort_values(by='Coeficiente', ascending=False)
        # Convertir el DataFrame a un string
        vRes += f"Coeficientes del modelo {pModelo.__class__.__name__}:\n"
        for idx, row in importancia_df.iterrows():
            vRes += f"Característica: {row['Característica']}, Coeficiente: {row['Coeficiente']}\n"


    # Verificar si el modelo es un árbol de decisión o un modelo basado en árboles (como RandomForest, GradientBoosting)
    elif hasattr(pModelo, 'feature_importances_'):
        # Para modelos basados en árboles, acceder a la importancia de las características
        importancia_features = pModelo.feature_importances_
        importancia_df = pd.DataFrame({
            'Característica': campos_dependientes,
            'Importancia': importancia_features
        }).sort_values(by='Importancia', ascending=False)

        # Convertir el DataFrame a un string
        vRes += f"Importancia de las características del modelo {pModelo.__class__.__name__}:\n"
        for idx, row in importancia_df.iterrows():
            vRes += f"Característica: {row['Característica']}, Importancia: {row['Importancia']}\n"

    else:
       vRes += f"El modelo {pModelo.__class__.__name__} no tiene coeficientes o importancias de características accesibles.\n"

    return vRes
