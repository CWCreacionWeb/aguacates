import sys
# Añadir el directorio de la clase
sys.path.append('/home/guille/UOC/aguacates/')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
# import ipywidgets as widgets
# from IPython.display import display
import random

class Charts:

    def __init__(self, file):
        data = pd.read_csv(file)
        self.df = pd.DataFrame(data)
        self.plt = plt
        self.sns = sns
        self.px = px
        self.np = np
        self.random = random
        self.pd = pd
        # Configuro el label de todas las graficas con los nombres de las regiones
        self.region_labels = self.df['region'].unique()
        self.seasonal_decompose = seasonal_decompose
        self.formatDate('Date')
        self.year = self.df['Date'].dt.year

    def figureConfig(self, width=11, height=6,**kwargs):
        plt.figure(figsize=(width, height))
        plt.legend(title='Region', labels=self.region_labels,bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7, which='both')
        # Configuramos las posicion de los textos de x
        plt.xticks(rotation=45)

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

    def show(self):
        plt.show()

    def plot(self, x, y, title, xlabel, ylabel,label=None,marker="o",show=True,**kwargs):
        plt.plot(x, y,marker=marker,label=label)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(title='Region', labels=self.region_labels,bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7, which='both')
        plt.xticks(rotation=45)
        plt.axhline(0, color='red', linestyle='--', label=label, linewidth=1)
        if kwargs:
            plt.gca().set(**kwargs)

        if show:
            plt.show()

    def plot_bar(self, x, y, title, xlabel, ylabel, ax=None, axis='both', alpha=0.75, ylim=None, show=True):
        if ax is None:
            fig, ax = plt.subplots()
        ax.bar(x, y, label=title)  # Añadir `label` para cada serie

        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.grid(axis=axis, alpha=alpha)
        if ylim:
            ax.set_ylim(ylim)
        if show:
            ax.legend(title="Tipo de Bolsa")  # Mostrar leyenda para identificar cada serie
            plt.show()

    def plot_hist(self, x, title, xlabel, ylabel,**kwargs):
        fig, ax = plt.subplots()
        ax.hist(x, bins=10)
        plt.axhline(0, color='red', linestyle='--', linewidth=1)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        if kwargs:
            ax.hist(x, bins=10, **kwargs)  # Pasa los kwargs a ax.hist


    def plot_box(self, x, title, xlabel, ylabel):
        fig, ax = plt.subplots()
        ax.boxplot(x)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.grid()
        plt.show()

    def temporada(self,fecha) :

        if fecha in [12, 1, 2]:
            return 'Invierno'
        elif fecha in [3, 4, 5]:
            return 'Primavera'
        elif fecha in [6, 7, 8]:
            return 'Verano'
        else:
            return 'Otoño'

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

        self.region_labels = self.df['region'].unique()


    def filtra(self, key, value):
        if key not in self.df.columns:
            raise ValueError(f"La columna '{key}' no existe en el DataFrame.")
        return self.df[self.df[key] == value]
