import json
from IPython.display import Markdown, display

class Enunciados:
    def __init__(self):
        # Cargar el JSON desde un archivo dentro de la carpeta 'datos'
        with open('./datos/enunciados.json', 'r') as file:
            self.enunciado = json.load(file)

    def getEnunciado(self, key):
        # Accede al enunciado específico
        enunciado = self.enunciado.get(key, {})
        descripcion = enunciado.get('descripcion', '')
        contenido = enunciado.get('contenido', '')
        comentarios = enunciado.get('comentarios', '')
        # Mostrar en formato Markdown
        display(Markdown(f"{key} | {descripcion}\n{contenido}\n{comentarios}"))

    def getExplicacion(self, key):
        # Accede al enunciado específico
        enunciado = self.enunciado.get(key, {})
        explicacion = enunciado.get('explicacion', '')
        # Mostrar en formato Markdown
        display(Markdown(f"{key} Explicación: | {explicacion}"))


    def get(self, key, default=None):
        return self.enunciado.get(key, default)