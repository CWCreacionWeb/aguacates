# Proyecto: # Análisis del Conjunto de Datos de Precios de Aguacate 


1. **Metodología de trabajo**
      - **Extreme programming (XP) :** Como un equipo de Jazz,
      - **Canales de comunicación :** whatsapp  y Google Meet video conferencia.
      - **Repositorio:** GitHub, cada usuario tiene un repositorio, con dos ramas para su organización main y DEV,para integrar los desarollos se realiza mediante Fork

# ** Descripción solución tecnica **
  De partida se utiliza Jupyter como interfaz de usuario y python para el desarrollo.
  A parte de los objetivos planteados en el proyecto, se desea que la solución sea escalable, flexible, inteligible. El planteamiento es el sigueinte
      - **Jupyter:** Solo se utilzia como visualizador, con celdas nada de texto.
      1. **Aplicación Python**
      .- Todo se desarrolla en ficheros de python.
      - Modulo principal,  ejecuta lo imprescindible e importa los modulos de codigo de codigo.
      - Como tareas previas y comunes a todos los modulos, añade columnas calculadas al DataFrame CalFecha,CalMonth, CalEstacion etc
      - Se crea un modulo para cada punto del proyeto las 2 primeras letras identifican el modulo P1, P2 etc
      - En cada modulo hay una fución por subpunto con la siguiente nomenclatura [Modulo]_[subpunto]_[Descripción] ejemplo def P1_1_DescomposicionSerieTemporal(pPeriodo=52,pCampo='AveragePrice'):
      - Documentación, cada función incluye la documentación del punto con formato Markdown copiada del original y la imprime como primer paso. Incluir parametros etc como aplicación.
      
    

