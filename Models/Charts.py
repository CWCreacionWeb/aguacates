import sys
# Añadir el directorio de la clase
sys.path.append('/home/guille/UOC/aguacates/')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
import ipywidgets as widgets
from IPython.display import display

class Charts:

    def __init__(self, file):
        data = pd.read_csv(file)
        self.df = pd.DataFrame(data)
        # Configuro el label de todas las graficas con los nombres de las regiones
        self.region_labels = self.df['region'].unique()
        self.seasonal_decompose = seasonal_decompose
        self.formatDate('Date')

    def figureConfig(self, width=11, height=6,**kwargs):
        plt.figure(figsize=(width, height))
        plt.legend(title='Region', labels=self.region_labels,bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7, which='both')
        # Uso kwargs directamente en plt para pasar etiquetas, leyendas, etc.
        if kwargs:
            plt.gca().set(**kwargs)
        plt.tight_layout()

    def showData(self,key = None):
        if key:
            return self.df[key]
        else:
            return self.df

    def selectedKeys(self, keys):
        return self.df[keys]

    def makeFrame(data):
        return pd.DataFrame(data)

    def showColumns(self):
        return self.df.columns

    def isNull(self):
        return self.df.isnull().sum()

    def clearData(self, column_name):
        self.df = self.df.drop(columns=[column_name])

    def formatDate(self, column_name):
        self.df[column_name] = pd.to_datetime(self.df[column_name])

    def plot(self, x, y):
        plt.plot(x, y,marker='o')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.show()

    def plot_bar(self, x, y, title, xlabel, ylabel):
        fig, ax = plt.subplots()
        ax.bar(x, y)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.grid()
        plt.show()

    def plot_pie(self, x, y, title):
        fig, ax = plt.subplots()
        ax.pie(y, labels=x, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        ax.set_title(title)
        plt.show()

    def plot_hist(self, x, title, xlabel, ylabel):
        fig, ax = plt.subplots()
        ax.hist(x, bins=10)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.grid()
        plt.show()

    def plot_box(self, x, title, xlabel, ylabel):
        fig, ax = plt.subplots()
        ax.boxplot(x)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.grid()
        plt.show()

    def temporada(fecha) :
        #COmprobamos a que estacion del año pertenece la fecha
        #obtenemos el mes de la fecha dada
        fecha = pd.to_datetime(fecha) #convertimos la fecha a formato datetime

        if fecha.month in [12, 1, 2]:
            return 'Winter'
        elif fecha.month in [3, 4, 5]:
            return 'Spring'
        elif fecha.month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'

    def paintPlotTitle(self, data,key1,key2,ha='center',va='bottom',fontsize=10,color="black"):
        for i in range(len(data)):
            plt.text(
                data[key1].iloc[i],
                data[key2].iloc[i],
                f'{data[key2].iloc[i]:.2f}',
                ha=ha,
                va=va,
                fontsize=fontsize,
                color=color
            )

    def to_excel(self, data, filename):
        # Guardar como archivo Excel
        excel_file = filename + '.xlsx'  # Nombre del archivo Excel a crear
        data.to_excel(excel_file, index=False)  # index=False para no incluir el índice como una columna

    # CREO una funcion para coger las 5 primeras regions con mas volumen de ventas sobvre el campo Total Volume
    def topRegions(self,num=5,exclude=None):
        if exclude:
            self.df = self.df[self.df['region'].isin(self.df.groupby('region')['Total Volume'].sum().sort_values(ascending=False).head(num).index)]
            self.df = self.df[self.df['region'] != exclude]
        else:
            self.df = self.df[self.df['region'].isin(self.df.groupby('region')['Total Volume'].sum().sort_values(ascending=False).head(num).index)]
