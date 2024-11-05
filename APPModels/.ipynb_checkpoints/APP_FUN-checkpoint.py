from APPModels.Enunciados import ClsEnunciados
from APPModels.Charts import Charts
APP_Enunciados =ClsEnunciados()

file = 'datos/avocado.csv'
chart = Charts(file)
# FILTRAMOS POR LAS 10 REGIONES CON MAS VOLUMEN DE VENTAS
chart.topRegions(num=10,exclude='TotalUS')
#Hacemos una limpieza la columna Unnamed
chart.clearData('Unnamed: 0')
# Comprobamos si hay valores nulos
chart.isNull()
chart.showData()
enun = APP_Enunciados
