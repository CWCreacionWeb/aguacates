import json
from IPython.display import Markdown, display

class ClsEnunciados:

    def __init__(self):
        print('ClsEnunciados__init__\n')
        # Cargar el JSON desde un archivo dentro de la carpeta 'datos'
        with open('./datos/enunciados.json', 'r') as file:
            self.enunciado = json.load(file)

    def getEnunciado(self, key,moreData=None):
        # Accede al enunciado específico
        enunciado = self.enunciado.get(key, {})
        descripcion = enunciado.get('descripcion', '')
        contenido = enunciado.get('contenido', '')
        comentarios = enunciado.get('comentarios', '')
        if moreData:
            contenido = contenido.format("\n**" + moreData)
        # Mostrar en formato Markdown
        if descripcion =="":
            display(Markdown(f"{contenido}"))
        else:
            display(Markdown(f"{key} | {descripcion}\n{contenido}\n{comentarios}"))

    def getExplicacion(self, key):
        # Accede al enunciado específico
        enunciado = self.enunciado.get(key, {})
        explicacion = enunciado.get('explicacion', '')
        # Mostrar en formato Markdown
        display(Markdown(f"{key} Explicación: | {explicacion}"))


    def get(self, key, default=None):
        return self.enunciado.get(key, default)

Enunciados =ClsEnunciados()
