import sys
# Añadir el directorio de la clase
sys.path.append('/home/guille/UOC/aguacates/')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose

class Charts:

    def __init__(self, file):
        data = pd.read_csv(file)
        self.df = pd.DataFrame(data)
        self.figureConfig()
        self.seasonal_decompose = seasonal_decompose

    def figureConfig(self, width=11, height=6,**kwargs):
        plt.figure(figsize=(width, height))
        plt.grid(True)
        # Uso kwargs directamente en plt para pasar etiquetas, leyendas, etc.
        if kwargs:
            plt.gca().set(**kwargs)

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


