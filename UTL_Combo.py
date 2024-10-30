# Importamos las librerías necesarias
from ipywidgets import widgets, VBox, HBox, Output, Button
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from IPython.display import display

class Widget_lst:
    def __init__(self, data, description, button_description, on_button_click=None):
        # Configuramos los parámetros iniciales del widget
        self.data = data
        self.description = description
        self.button_description = button_description
        self.on_button_click = on_button_click  # Función personalizada para manejar el clic
        
        
        # Creamos el widget de salida y el widget de selección múltiple
        self.salida = Output()
        self.wLista_widgets = widgets.SelectMultiple(
            options=self.data,
            description=self.description,
            disabled=False
        )
        
        # Creamos el botón y asociamos la función de ejecución
        self.wBtn_Lista = Button(description=self.button_description)
        self.wBtn_Lista.on_click(self.Btn_Lista_Ejecutar)

    def Btn_Lista_Ejecutar(self, b):
        # Limpiamos la salida para que no se acumulen resultados previos
        self.salida.clear_output()
        with self.salida:
            # Obtenemos las opciones seleccionadas y las mostramos
            seleccion = list(self.wLista_widgets.value)
            # Si se ha pasado una función personalizada, la llamamos con la selección
            if self.on_button_click:
                self.on_button_click(seleccion)
            else:
                # Si no hay función personalizada, imprimimos la selección por defecto
                print("Opciones seleccionadas:", seleccion)
                
    def mostrar(self):
        # Mostramos el widget de selección, el botón y el área de salida
        display(VBox([self.wLista_widgets, self.wBtn_Lista, self.salida]))


"""
# Creamos un widget de salida para mostrar los resultados
salida = Output()


# Creamos un widget de lista multiselección
WLista_Data=[]
WLista_description='Sin Asignar WLista_description'
WLista_description_Button='Sin Asignar WLista_description_Button'

wLista_widgets = None  # Inicializamos como None
wBtn_Lista = None

def Inicio():
    global wBtn_Lista
    global wLista_widgets
    wLista_widgets = widgets.SelectMultiple(
        options=WLista_Data,
        description=WLista_description,
        disabled=False
    )
    # Creamos un botón para la acción
    wBtn_Lista = Button(description=WLista_description_Button)
    # Función que se ejecutará cuando el botón sea presionado
    #display(VBox([wLista_widgets, wBtn_Lista, salida]))
    # Asociamos la función al botón
    wBtn_Lista.on_click(Btn_Lista_Ejecutar)
    display(VBox([wLista_widgets, wBtn_Lista, salida]))

def Btn_Lista_Ejecutar(b):
    # Limpiamos la salida para que no se acumulen gráficos previos
    salida.clear_output()
    with salida:
        # Obtenemos las ciudades seleccionadas
        seleccion = list(wLista_widgets.value)
        print(seleccion)


#def W_Display():
#    display(VBox([wLista_widgets, wBtn_Lista, salida]))
"""

