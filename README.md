# Evaluacion_IA-U2
Aplicación de un algoritmo de machine learning a un dataset


Integrantes:
* Adriana Conde
* Paul Pasaca
* Anderson Quizhpe
* Luis Negron

Ciclo: 9no A
Asignatura: Inteligencia Artificial

# Herramientas
## Weka
Es un software de código abierto que proporciona herramientas para el preprocesamiento de datos, la implementación de varios algoritmos de aprendizaje automático y herramientas de visualización para que pueda desarrollar técnicas de aprendizaje automático y aplicarlas a problemas de minería de datos del mundo real. [1] Puede descargar la aplicaión desde el siguiente enlace: https://waikato.github.io/weka-wiki/downloading_weka/
## Python
Python es un popular lenguaje de programación orientado a objetos que tiene las capacidades de un lenguaje de programación de alto nivel. Características de Python que lo convierten una opción para el aprendizaje automátio [2]:
- Amplio conjunto de paquetes como numpy, scipy, pandas, scikit-learn, etc., que son necesarios para el aprendizaje automático.
- Python es un lenguaje que permite la creación de prototipos fácil y rápida
- Incluye varios dominios como extracción de datos, manipulación de datos, análisis de datos, extracción de características, modelado, evaluación, implementación.


## Google Colab
Es un producto de Google Research. Permite a cualquier usuario escribir y ejecutar código arbitrario de Python en el navegador. Es especialmente adecuado para tareas de aprendizaje automático, análisis de datos y educación. Desde un punto de vista más técnico, Colab es un servicio alojado de Jupyter Notebook que no requiere configuración y que ofrece acceso gratuito a recursos informáticos, como GPUs.[3]

# Algoritmos Utilizados
## Arboles de Decisión
Un Árbol de Decisión es un método analítico que a través de una representación esquemática de las alternativas disponibles facilita la toma de mejores decisiones, especialmente cuando existen riesgos, costos, beneficios y múltiples opciones. El nombre se deriva de la apariencia del modelo parecido a un árbol y su uso es amplio en el ámbito de la toma de decisiones bajo incertidumbre.  [4] 
#### Estructura básica de un árbol de decisión
Los árboles de decisión están formados por nodos y su lectura se realiza de arriba hacia abajo. 
Dentro de un árbol de decisión distinguimos diferentes tipos de nodos: 
* Primer nodo o nodo raíz: en él se produce la primera división en función de la variable más importante.
* Nodos internos o intermedios: tras la primera división encontramos estos nodos, que vuelven a dividir el conjunto de datos en función de las variables.
* Nodos terminales u hojas: se ubican en la parte inferior del esquema y su función es indicar la clasificación definitiva

#### Ventajas
- Son fáciles de construir, interpretar y visualizar.
- Selecciona las variables más importantes y en su creación no siempre se hace uso de todos los predictores.
- Si faltan datos no podremos recorrer el árbol hasta un nodo terminal, pero sí podemos hacer predicciones promediando las hojas del sub-árbol que alcancemos.
- No es preciso que se cumplan una serie de supuestos como en la regresión lineal (linealidad, normalidad de los residuos, homogeneidad de la varianza, etc.).
- Sirven tanto para variables dependientes cualitativas como cuantitativas, como para variables predictoras o independientes numéricas y categóricas. Además, no necesita variables dummys, aunque a veces mejoran el modelo.
- Permiten relaciones no lineales entre las variables explicativas y la variable dependiente.
- Nos podemos servir de ellos para categorizar variables numéricas.[4]

#### Desventajas

- Tienden al sobreajuste u overfitting de los datos, por lo que el modelo al predecir nuevos casos no estima con el mismo índice de acierto.
- Se ven influenciadas por los outliers, creando árboles con ramas muy profundas que no predicen bien para nuevos casos. Se deben eliminar dichos outliers.
- No suelen ser muy eficientes con modelos de regresión.
- Crear árboles demasiado complejos puede conllevar que no se adapten bien a los nuevos datos. La complejidad resta capacidad de interpretación.
- Se pueden crear árboles sesgados si una de las clases es más numerosa que otra.
- Se pierde información cuando se utilizan para categorizar una variable numérica continua. [4]

### Algoritmo J48
### Decision Stump
### LMT
### Random Tree
## Clústers
### Kmeans
El algoritmo de agrupación en clústeres de K-Means se define como un método de aprendizaje no supervisado que tiene un proceso iterativo en el que el conjunto de datos se agrupa en un número k de clústeres o subgrupos predefinidos que no se superponen, haciendo que los puntos internos del clúster sean lo más similares posible mientras se intenta mantener el agrupa en un espacio distinto.

Se utiliza cuando tenemos un montón de datos sin etiquetar. El objetivo de este algoritmo es el de encontrar “K” grupos (clusters) entre los datos crudos. 
Los “centroides” de cada grupo que serán unas “coordenadas” de cada uno de los K conjuntos que se utilizarán para poder etiquetar nuevas muestras.
Etiquetas para el conjunto de datos de entrenamiento. Cada etiqueta perteneciente a uno de los K grupos formados

# Aplicación del Algoritmo
## Arboles de Decision
1. Abrimos la herramienta weka y seleccionamos la opcion Open File para poder cargar nuestro dataset.
<!--Imagen--> 
![image](https://raw.githubusercontent.com/andersonquizhpe/colabIA/main/imagenesML/abrirdata.png)
2.  Una vez abierto el dataset, nos dirigimos a la pestaña de Clasify
3.  En la opción Choose escogemos el algoritmo que utilizaremos
4.  Seleccionamos la variable que nos permitira clasificar los datos
![image](https://raw.githubusercontent.com/andersonquizhpe/colabIA/main/imagenesML/Segundopaso.png)

5.  jb

# Conclusión
- 
# Bibliografía
- [1] Tutorials Point, “What is Weka?”. Disponible en: https://www.tutorialspoint.com/weka/what_is_weka.htm (consultado Feb. 19, 2022).
- [2] Tutorials Point, Machine Learning with Python. Disponible en: https://www.tutorialspoint.com/machine_learning_with_python/machine_learning_with_python_tutorial.pdf.
- [3] Google, “Google Colab.” Disponible en: https://research.google.com/colaboratory/intl/es/faq.html (consultado Feb. 19, 2022).
- [4] Qué son los árboles de decisión y para qué sirven | Máxima Formación. (n.d.). Retrieved February 13, 2022, from https://www.maximaformacion.es/blog-dat/que-son-los-arboles-de-decision-y-para-que-sirven/
