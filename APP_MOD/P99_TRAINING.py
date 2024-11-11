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
from APPModels.APP_FUN import APP_Enunciados
import APPModels.APP_FUN as app_fun  # Importa el módulo completo


from scipy.optimize import minimize
from itertools import combinations
from IPython.display import clear_output, display
import joblib

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



class ModeloManager:
    def __init__(self):
        """
        Inicializa un gestor para los modelos y sus metadatos.
        """
        self.registros = []  # Lista de registros (diccionarios)
    def exportar_a_excel(self, nombre_archivo="registros_modelos.xlsx"):
        """
        Exporta todos los registros a un archivo Excel (.xlsx).

        Parámetro:
        nombre_archivo (str): El nombre del archivo Excel a guardar.
        """
        if not self.registros:
            print("No hay registros para exportar.")
            return

        # Convertir la lista de registros en un DataFrame
        df_registros = pd.DataFrame(self.registros)

        # Guardar el DataFrame en un archivo Excel
        df_registros.to_excel(nombre_archivo, index=False)
        print(f"Registros exportados a: {nombre_archivo}")        

    def actualizar_txt_coef(self, nuevo_txt_coef):
        """
        Actualiza el valor de 'TXT_Coef' en el último registro agregado.

        Parámetro:
        nuevo_txt_coef (str): El nuevo valor para el campo 'TXT_Coef'.
        """
        # Actualizamos el 'TXT_Coef' en el último registro
        self.registros[-1]['TXT_Coef'] = nuevo_txt_coef
    
    def ejecutar_modelo(self,  modelo, X_train, y_train, X_test, y_test):
        """
        Agrega un nuevo registro de modelo con sus metadatos al gestor.

        Parámetros:
        nombre_modelo (str): El nombre del modelo (como 'LinearRegression' o 'GBR').
        modelo (objeto de modelo): El modelo a entrenar (como LinearRegression(), GradientBoostingRegressor()).
        X_train (numpy.array): Los datos de entrada (campos independientes).
        y_train (numpy.array): Los datos objetivo (campo dependiente).
        """
        start_time = time.time()  # Iniciar temporizador
        
        # Entrenar el modelo
        modelo.fit(X_train, y_train)
        
        # Evaluación en el conjunto de entrenamiento
        y_train_pred = modelo.predict(X_train)
        mse_train = mean_squared_error(y_train, y_train_pred)
        rmse_train = np.sqrt(mse_train)
        r2_train = r2_score(y_train, y_train_pred)

        # Evaluación en el conjunto de prueba
        y_test_pred = modelo.predict(X_test)
        mse_test = mean_squared_error(y_test, y_test_pred)
        rmse_test = np.sqrt(mse_test)
        r2_test = r2_score(y_test, y_test_pred)
        
        # Medir el tiempo de ejecución
        tiempo_ejecucion = time.time() - start_time

        
        # Crear un diccionario con el nuevo registro
        registro = {
            'NombreModelo': modelo.__class__.__name__,
            'Modelo': modelo,  # Aquí se guarda el objeto del modelo
            'CampoDependiente': y_train.name ,
            'CamposIndependientesNum': len(X_train.columns),
            'CamposIndependientes': list(X_train.columns),
            'tiempo_ejecucion': tiempo_ejecucion,
            'MSE_train': mse_train,
            'RMSE_train': rmse_train,
            'R2_train': r2_train,
            'MSE_test': mse_test,
            'RMSE_test': rmse_test,
            'R2_test': r2_test,
        }
        
        # Añadir el registro a la lista de registros
        self.registros.append(registro)
    
    def grabar_registros(self, nombre_archivo):
        """
        Graba todos los registros en un archivo usando joblib.

        Parámetro:
        nombre_archivo (str): El archivo donde guardar los registros persistidos.
        """
        joblib.dump(self.registros, nombre_archivo)
        print(f"Registros guardados en: {nombre_archivo}")
    
    def cargar_registros(self, nombre_archivo):
        """
        Carga los registros desde un archivo.

        Parámetro:
        nombre_archivo (str): El archivo desde donde cargar los registros.
        """
        self.registros = joblib.load(nombre_archivo)
        print(f"Registros cargados desde: {nombre_archivo}")
    
    def acceder_registro(self, indice):
        """
        Accede a un registro específico por su índice.

        Parámetro:
        indice (int): El índice del registro que quieres acceder.

        Retorna:
        dict: El diccionario del registro solicitado.
        """
        if indice < 0 or indice >= len(self.registros):
            print("Índice fuera de rango.")
            return None
        return self.registros[indice]
    
    def mostrar_todos_los_registros(self):
        """
        Muestra todos los registros de los modelos utilizando la función info.
        """
        for i in range(len(self.registros)):
            informacion = self.info(i)  # Usar el método info para obtener la información formateada
            print(f"Registro {i+1}: {informacion}")


    def info(self, indice):
        """
        Accede a la información de un registro específico y la retorna como un string sin saltos de línea.
        
        Parámetro:
        indice (int): El índice del registro que quieres acceder.

        Retorna:
        str: Información del registro formateada.
        """
        if indice < 0 or indice >= len(self.registros):
            print("Índice fuera de rango.")
            return None
        
        modelo_record = self.registros[indice]
        
        # Formatear los valores numéricos a 3 decimales
        tiempo_formateado = f"{modelo_record.tiempo_ejecucion:.3f}"
        mse_formateado = f"{modelo_record.mse:.3f}"
        rmse_formateado = f"{modelo_record.rmse:.3f}"
        r2_formateado = f"{modelo_record.r2:.3f}"
        
        # Formatear los valores numéricos a 3 decimales
        tiempo_formateado = f"{modelo_record['tiempo_ejecucion']:.3f}"
        mse_formateado = f"{modelo_record['MSE_train']:.3f}"
        rmse_formateado = f"{modelo_record['RMSE_train']:.3f}"
        r2_formateado = f"{modelo_record['R2_train']:.3f}"
        
        # Construir el string de salida con toda la información
        info = (
            f"NombreModelo: {modelo_record['NombreModelo']}, "
            f"CamposIndependientesNum: {modelo_record['CamposIndependientesNum']}, "
            f"tiempo_ejecucion: {tiempo_formateado}, "
            f"MSE_train: {mse_formateado}, "
            f"RMSE_train: {rmse_formateado}, "
            f"R2_train: {r2_formateado}, "
            f"CampoDependiente: {modelo_record['CampoDependiente']}, "
            f"CamposIndependientes: {modelo_record['CamposIndependientes']}, "
            f"TXT_Coef: {modelo_record['TXT_Coef']}"  # Incluir el campo TXT_Coef
        )        
        return info        


gestor_modelos = ModeloManager()

def P99_1_Modelo_TRAINING_TODOS():
        # Lista de modelos base a utilizar
    global mDbg
    mDbg =''
    modelos = [
        LinearRegression(),
        Lasso(alpha=0.1),
        Ridge(alpha=1.0),
        ElasticNet(alpha=0.1, l1_ratio=0.5),
        DecisionTreeRegressor(random_state=42),
        RandomForestRegressor(n_estimators=100, random_state=42),
        GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
        SVR(C=1.0, epsilon=0.1),
        KNeighborsRegressor(n_neighbors=5),
        XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    ]
    
    # Iterar sobre cada modelo y ejecutar la función con el modelo correspondiente
    for modelo_base in modelos:
        P99_1_Modelo_TRAINING(modelo_base)
    gestor_modelos.grabar_registros('P00_TRAINING_RES')
    gestor_modelos.exportar_a_excel()

def P100_1_Modelo_TRAINING_Mod(pAddColum=False):
    vModelo =LinearRegression()
    # Mostrar el enunciado
    SubDatos = app_fun.APP_DatosORG.copy()
        # Definir las columnas independientes y dependientes
    campo_independiente = "AveragePrice"
    #campos_dependientes = ["Total Volume", "4046", "4225", "4770", "Total Bags", "Small Bags", "Large Bags", "XLarge Bags", "Cal_AAAAMM", "Cal_AAAA", "Cal_MM","Cal_SS","Cal_DDD","Cal_AAAADDD","CalNOR_TotalVolume"]
    campos_dependientes = ["CalNOR_TotalVolume",   "Cal_AAAA", "Cal_MM"]

        # Separar los datos en características (X) y objetivo (y)
    X = SubDatos[campos_dependientes]
    y = SubDatos[campo_independiente]
    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entrenar el modelo
    vModelo.fit(X_train, y_train)
        
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
        
    vCoef = ''
    vCoef = P99_1_Modelo_TRAINING_FIT_Coef(vModelo,campo_independiente,campos_dependientes)

    # Aplicar el modelo a los datos org
    app_fun.APP_DatosORG[campo_independiente + '_PREV'] = vModelo.predict(SubDatos[campos_dependientes])



def P99_1_Modelo_TRAINING(pModelo,pAddColum=False):

    global mDbg
    vDbg =""
    vResumen =""
    vAvance =""
    nombre_modelo = pModelo.__class__.__name__

    # Mostrar el enunciado
    SubDatos = app_fun.APP_DatosORG.copy()
        # Definir las columnas independientes y dependientes
    campo_independiente = "AveragePrice"
    #campos_dependientes = ["Total Volume", "4046", "4225", "4770", "Total Bags", "Small Bags", "Large Bags", "XLarge Bags", "Cal_AAAAMM", "Cal_AAAA", "Cal_MM","Cal_SS","Cal_DDD","Cal_AAAADDD","CalNOR_TotalVolume"]

    campos_dependientes = ["Total Volume","CalNOR_TotalVolume",  "Cal_AAAAMM", "Cal_AAAA", "Cal_MM","Cal_SS","Cal_DDD","Cal_AAAADDD"]

        # Crear un DataFrame para almacenar resultados
    resultados = pd.DataFrame(columns=["campos_actuales", "error", "tiempo_ejecucion"])

    # Variables para el seguimiento del error
    error_minimo = float('inf')
    error_maximo = float('-inf')
    campos_error_minimo = None
    campos_error_maximo = None

    # Recorrer todas las combinaciones de los campos dependientes
    for i in range(1, len(campos_dependientes) + 1):
        for combinacion in combinations(campos_dependientes, i):
            # Convertir la combinación a lista para pasarlo como argumento
            campos_actuales = list(combinacion)
            clear_output(wait=True)  # Limpia la celda antes de mostrar la nueva salida
            # Llamar a la función de entrenamiento del modelo con la combinación actual
            P99_1_Modelo_TRAINING_FIT_Lineal(pModelo,SubDatos, campo_independiente, campos_actuales)



            #display(campos_actuales)
            display(len(gestor_modelos.registros))

    mDbg += f"{pModelo.__class__.__name__} Error  {error_minimo:.3f} a {error_maximo:.3f} con campos {campos_error_minimo}<br>"
    display(HTML(  mDbg))
    # Ordenar los resultados por error (MSE) de manera descendente
    resultados_ordenados = resultados.sort_values(by="error", ascending=False)

    gestor_modelos.grabar_registros('modelos_training')
    gestor_modelos.exportar_a_excel()
    time.sleep(0.5)  # Añadir una pequeña pausa para ver la actualización (opcional)

    # Generar el nombre del archivo con el formato [nombre del fichero]_AAAAMMDDHHMM.xlsx
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    nombre_fichero = f"xls/P5_3_TRAINING_{nombre_modelo}{timestamp}.xlsx"

    # Guardar el DataFrame ordenado en un archivo Excel con el nombre generado
    resultados_ordenados.to_excel(nombre_fichero, index=False)
    print(f"Archivo Excel guardado como '{nombre_fichero}'")

def P99_1_Modelo_TRAINING_FIT_Lineal(pModelo,df, campo_independiente, campos_dependientes, test_size=0.2):
        # Separar los datos en características (X) y objetivo (y)
    X = df[campos_dependientes]
    y = df[campo_independiente]
    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # Agregar el registro del modelo al gestor
    gestor_modelos.ejecutar_modelo(pModelo, X_train, y_train, X_test, y_test)

    vCoef = ''
    vCoef = P99_1_Modelo_TRAINING_FIT_Coef(pModelo,campo_independiente,campos_dependientes)
    gestor_modelos.actualizar_txt_coef(vCoef)
    #print(f"Error cuadrático medio (MSE): {error}")

def P99_1_Modelo_TRAINING_FIT_Coef(pModelo, campo_independiente, campos_dependientes):
        # 6. Mostrar los pesos de las características
    vRes =''
    # Verificar si el modelo tiene el atributo 'coef_' (tipico de modelos lineales)
    if hasattr(pModelo, 'coef_'):

        coeficientes_features  = pModelo.coef_
        importancia_df = pd.DataFrame({
            'Característica': campo_independiente,
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
 
