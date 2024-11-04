from Models.Charts import Charts
class P_1:

    def p1_1(self):
        """
        Descomposición de Series Temporales de Precios:
        Uso de Datos: Usa la columna AveragePrice y Date.
        Esperado: Utiliza la función seasonal_decompose de la librería statsmodels para descomponer la serie temporal de precios en componentes de tendencia, estacionalidad y ruido.
        Convierte Date a tipo datetime usando pd.to_datetime().
        Agrupa los datos por Date y calcula el promedio de AveragePrice utilizando groupby() si es necesario.
        Visualiza los componentes descompuestos usando matplotlib para cada uno de ellos.
        """
        file = 'datos/avocado.csv'
        chart = Charts(file)
        #Hacemos una limpieza la columna Unnamed
        chart.clearData('Unnamed: 0')
        # Comprobamos si hay valores nulos
        chart.isNull()
        chart.showData()
