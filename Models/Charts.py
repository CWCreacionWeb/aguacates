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
        plt.figure(figsize=(10, 5))
        self.seasonal_decompose = seasonal_decompose

    def figureConfig(self, width=12, height=6):
        plt.figure(figsize=(width, height))

    def showData(self):
        return self.df

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

    def plot(self, x, y, title=None, xlabel=None, ylabel=None):
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
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

