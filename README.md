# Proyecto: # Análisis del Conjunto de Datos de Precios de Aguacate 

Requerimientos [Proyecto_Analisis_Aguacate.pdf](/documentacion/Proyecto_Analisis_Aguacate.pdf) 

## Metodología de trabajo

- **Extreme programming (XP) :** Como un equipo de Jazz,
- **Trello :** Para gestionar la lista de tareas mediante KANBAN 
- **Canales de comunicación equpipo :** whatsapp  y Google Meet video conferencia.
- **Repositorio:** GitHub, cada usuario tiene un repositorio, con dos ramas para su organización main y DEV,para integrar los desarollos se realiza mediante Fork  
Este enfoque no ha sido practico y agil en el ejercicio y metos todavía para la gestión correcta de versiónes y una intregración que garantice la calidad en el repositorio principal  
\ncon un entorno (o rama) de test e incluso otro de QA solucionaría el problema de calidad.  
el costo de esta gestión no es menospreciable, el costo final que genera un error (tiempo perdido por los usuarios,feeling del cliente/usuario etc) puede y suele ser superior al de gestión de un entorno de test y QA.
## Descripción solución tecnica
  De partida se utiliza Jupyter como interfaz de usuario y python para el desarrollo.
  A parte de los objetivos planteados en el proyecto, se desea que la solución sea escalable, flexible, inteligible. 
  El planteamiento es el sigueinte
- **Jupyter** - Solo se utilzia como visualizador, con celdas, nada de texto.
- **Aplicación Python** Solo se utilzia como visualizador, con celdas, nada de texto.
   - Todo se desarrolla en ficheros de python.
  - **Modulo principal**:  ejecuta lo imprescindible e importa los modulos de codigo de codigo.
  - **Preparación de datos**: Como tareas previas y comunes a todos los modulos, añade columnas calculadas al DataFrame CalFecha,CalMonth, CalEstacion etc 
  - **Modulos**:Se crea un modulo para cada punto del proyeto las 2 primeras letras identifican el modulo `P1`, `P2` etc
  - En cada modulo hay una fución por subpunto con la siguiente nomenclatura `[Modulo]_[subpunto]_[Descripción]` ejemplo `def P1_1_DescomposicionSerieTemporal(pPeriodo=52,pCampo='AveragePrice'):`
  - **Documentación**, cada función incluye la documentación del punto con formato `Markdown` copiada del original y la imprime como primer paso. Incluir parametros etc como aplicación.

      
## Como usar la aplicación

En un fichero de jupyter, ejecutar los procesos en las celdas.

**Primer paso Importar el modulo y ejecutar los procesos imprescindibles :**


```sh
import P_Aguacate as P
```
```sh
P.Inicio()
```

**Ejemplo de codigo para ejecutar cualquier punto del proyecto :**
Como puedes observar `P` es el Proyecto importado, `P1` es el modulo , y `P1_1_DescomposicionSerieTemporal(52,'AveragePrice')` la función con los parametros.
```sh
P.P1.P1_1_DescomposicionSerieTemporal(52,'AveragePrice')
```
![Readme ejemplo 01](/documentacion/Readme_Ejemplo.png)

## Links

Editor `Markdown Live Preview` https://markdownlivepreview.com/


