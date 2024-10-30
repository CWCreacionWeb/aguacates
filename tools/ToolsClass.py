import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose


class tools:

    def __init__(self,marker,color):
        self.color = color
        self.marker = marker

    def date_format(date):
        return pd.to_datetime(date)

    def p(self):
        return 'HOLA'