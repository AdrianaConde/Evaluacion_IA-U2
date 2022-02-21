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

# Algoritmos Utilizados
## Arboles de Decisión
Un Árbol de Decisión es un método analítico que a través de una representación esquemática de las alternativas disponibles facilita la toma de mejores decisiones, especialmente cuando existen riesgos, costos, beneficios y múltiples opciones. El nombre se deriva de la apariencia del modelo parecido a un árbol y su uso es amplio en el ámbito de la toma de decisiones bajo incertidumbre.  [3] 
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
- Nos podemos servir de ellos para categorizar variables numéricas.[3]

#### Desventajas

- Tienden al sobreajuste u overfitting de los datos, por lo que el modelo al predecir nuevos casos no estima con el mismo índice de acierto.
- Se ven influenciadas por los outliers, creando árboles con ramas muy profundas que no predicen bien para nuevos casos. Se deben eliminar dichos outliers.
- No suelen ser muy eficientes con modelos de regresión.
- Crear árboles demasiado complejos puede conllevar que no se adapten bien a los nuevos datos. La complejidad resta capacidad de interpretación.
- Se pueden crear árboles sesgados si una de las clases es más numerosa que otra.
- Se pierde información cuando se utilizan para categorizar una variable numérica continua. [3]

### Algoritmo J48
Es uno de los algoritmos de minería de datos más utilizados. El algoritmo del árbol de decisión es para averiguar la forma en que se comporta el vector de atributos para una serie de instancias, este algoritmo genera un árbol de decisión injertado el cual es utilizado en esencia para reducir errores de predicción al momento de generar un modelo. [4] [5] 

### Decision Stump
El operador Decision Stump se utiliza para generar un árbol de decisión con una única división. El árbol resultante se puede utilizar para clasificar ejemplos no vistos.  Los nodos de hoja de un árbol de decisión contienen el nombre de la clase, mientras que un nodo que no es de hoja es un nodo de decisión. El nodo de decisión es una prueba de atributo en la que cada rama (a otro árbol de decisión) es un valor posible del atributo. [6] [7]

### LMT
Los árboles modelo logísticos se basan en la idea anterior de un árbol modelo: un árbol de decisión que tiene modelos de regresión lineal en sus hojas para proporcionar un modelo de regresión lineal por partes, es decir, los árboles de decisión ordinarios con constantes en sus hojas producirían un modelo constante por partes [8]
### Random Tree
Random Tree es un Clasificador supervisado; es un algoritmo de aprendizaje que genera conjuntos de aprendizajes individuales. Los árboles aleatorios son un grupo de predictores de árboles que se denomina bosque. Los mecanismos de clasificación son los siguientes: el clasificador de árboles aleatorios obtiene el vector de características de entrada, lo clasifica con cada árbol del bosque y genera la etiqueta de clase que recibió la mayoría de los "votos". Los árboles aleatorios son esencialmente la combinación de dos algoritmos existentes en el aprendizaje automático: los árboles modelo únicos se fusionan con modelos de bosques aleatorios.[9]

# Comparación entre algoritmos
![comparacion](https://user-images.githubusercontent.com/40923800/154867575-7390d553-62d9-4376-9d43-f168fd37861c.JPG)

Como podemos observar en la tabla anterior, entre los algoritmos que clasifican con mayor precision se encuentra los algoritmos de RandomTree y el algoritmo J48, por lo que se decidio usar el algoritmo Random tree

## Clústers
### Kmeans
El algoritmo de agrupación en clústeres de K-Means se define como un método de aprendizaje no supervisado que tiene un proceso iterativo en el que el conjunto de datos se agrupa en un número k de clústeres o subgrupos predefinidos que no se superponen, haciendo que los puntos internos del clúster sean lo más similares posible mientras se intenta mantener el agrupa en un espacio distinto.

Se utiliza cuando tenemos un montón de datos sin etiquetar. El objetivo de este algoritmo es el de encontrar “K” grupos (clusters) entre los datos crudos. 
Los “centroides” de cada grupo que serán unas “coordenadas” de cada uno de los K conjuntos que se utilizarán para poder etiquetar nuevas muestras.
Etiquetas para el conjunto de datos de entrenamiento. Cada etiqueta perteneciente a uno de los K grupos formados


### Criterio de parada para K-Means Clustering

Existen tres criterios para terminar el algoritmo:

- Los centroides dejan de cambiar. Es decir, después de múltiples iteraciones, los centroides de cada clúster no cambian. Por lo que se asume que el algoritmo ha convergido.
- Los puntos dejan de cambiar de clúster. Parecido al anterior, pero esta vez no nos fijamos en los centroides, si no en los puntos que pertenecen a cada clúster. Cuando se observa que no hay un intercambio de clústers se asume que el modelo está entrenado.
- Límite de iteraciones. Podemos fijar un número máximo de iteraciones que queremos que nuestro algoritmo ejecute antes de pararlo. Cuando llega a ese número, se asume que el modelo no va a mejorar drásticamente y se para el entrenamiento.

### Desventajas de K-Means

Ya hemos visto la potencia que tiene este algoritmo. Por lo sencillo que es de aplicar y la valiosa información sobre nuestros datos que nos aporta. Como no es oro todo lo que reluce, tengo que comentaros también las desventajas que ofrece:

- Tenemos que elegir k nosotros mismos. Es muy posible que nosotros cometamos un error, o que sea imposible escoger una k óptima.
- Es sensible a outliers. Los casos extremos hacen que el clúster se vea afectado. Aunque esto puede ser algo positivo a la hora de detectar anomalías.
- Es un algoritmo que sufre de la maldición de la dimensionalidad. 

### Ventajas de K-Means

- Prácticamente funcionan bien, incluso algunos supuestos se rompen
- Simple, fácil de implementar
- Fácil de interpretar los resultados de agrupamiento
- Rápido y eficiente en términos de costo computacional, típicamente O (K * n * d)

# Comparacion de Weka y Python
![ksnip_20220220-204018](https://user-images.githubusercontent.com/40868390/154875469-547283e7-a2ae-4195-bc4f-07780d00d353.png)

# Aplicación de los Algoritmos
## Arboles de Decision
1. Abrimos la herramienta weka y seleccionamos la opcion Open File para poder cargar nuestro dataset.
<!--Imagen--> 
![abrirdata](https://user-images.githubusercontent.com/40923800/154828958-4e31c28f-c6be-44f5-8cf7-518dbb8dca08.png)

2.  Una vez abierto el dataset, nos dirigimos a la pestaña de Clasify
3.  En la opción Choose escogemos el algoritmo que utilizaremos, para este caso utilizaremos el algoritmo random tree
4.  Seleccionamos la variable que nos permitira clasificar los datos
![Segundopaso](https://user-images.githubusercontent.com/40923800/154828995-ac41ec53-0d04-4db0-a57a-b966f04f54c9.png)

5. Definimos los parametros para la aplicación del algoritmo
6. Entrenamos al algoritmo con la data que seleccionamos al inicio
![Captura2](https://user-images.githubusercontent.com/40923800/154865002-44601dcb-c7b9-44c0-8c86-d18f1c6bbf0c.JPG)
7. Una vez entrenado el modelo se presentan los resultados de la evaluacion del mismo, asi como la precisión, que para este caso nos presenta el 100% de clasificación correcta
![Captura 3JPG](https://user-images.githubusercontent.com/40923800/154865149-583c0536-668b-4284-91c3-c1e60d4e9e0c.JPG)
A traves del mismo weka podemos visualizar el arbol de decision que se genero del entrenamiento
![arbol](https://user-images.githubusercontent.com/40923800/154866044-c04ff258-53e1-4e85-b60a-be0cc2bf8cab.JPG)
Además de esto, weka nos permite obtener un grafico en el cual podemos observar la probabilidad de que un dato sea clasificado correctamente para esto hacemos click derecho y elgimos Visualize margin curve
![jkjnkm](https://user-images.githubusercontent.com/40923800/154866441-c7827c74-671d-430c-a7bf-9a0e599aa03e.jpg)

Finalmente obtenemos la grafica con los datos que se mencion, en este caso la probabilidad es igual a 1
![proba](https://user-images.githubusercontent.com/40923800/154866491-ca574376-8b3f-47ae-9b71-03e8488f6872.JPG)

## Predicción
8. Ya con el modelo entrenado podemos predecir nuevos datos, para ello elegimos el nuevo dataset
9. El nuevo dataset tendra un campo adicional el mismo que nos servira para la predicción
10. Elegimos el campo que se desea predecir, en este caso utilizaremos label
![wwCaptura](https://user-images.githubusercontent.com/40923800/154865476-744819aa-5c39-4a13-8689-378093f842f9.JPG)
11. Elegimos la forma en que deseamos que se presente los datos
![Captura](https://user-images.githubusercontent.com/40923800/154865767-a41f7d21-dfcf-4725-87f3-ab12c2920c53.JPG)
12. Finalmente se observa la prediccion realizada con los nuevos datos
![CapturaFinal](https://user-images.githubusercontent.com/40923800/154865934-b596fc49-a783-47c9-9014-569618d5e07a.JPG)

## K-Means

1. Abrimos el programa Weka y cargamos nuestro dataset con el boton open file
![ksnip_20220220-191326](https://user-images.githubusercontent.com/40868390/154871003-343c6762-ce68-498c-ba24-bad37e3d8219.png)

2. Eliminamos las columnas que no contienen datos numericos, ya que el metodo solo trabaja con numeros
![ksnip_20220220-192936](https://user-images.githubusercontent.com/40868390/154871600-5d5742f6-2e47-449d-9f70-e7b35ac68077.png)

3. Nos dirigimos a la pestaña cluster en la parte superior

4. Escogemos el algoritmo que deseamos usar con el boton choose, en este caso escogeremos el simple kmeans
![ksnip_20220220-193502](https://user-images.githubusercontent.com/40868390/154871844-5931cb65-5f54-4ef5-99f6-e9fe1085a442.png)

5. Colocamos los parametros para aplicar el algoritmo 
![ksnip_20220220-193806](https://user-images.githubusercontent.com/40868390/154871978-e26d6203-5225-4c15-bb18-7071270f6754.png)

6. Presionamos el boton start para empezar el entrenamiento
![ksnip_20220220-194020](https://user-images.githubusercontent.com/40868390/154872084-7b078769-1c08-490e-b344-8897205c9d51.png)

7. Luego de que el modelo se entrene se podra obserbar los siguientes resultados
En la imagen podemos observar el numero de iteraciones realizado, tambien el valor de los centroides calculados
![ksnip_20220220-194335](https://user-images.githubusercontent.com/40868390/154872243-dc095d42-6d8e-4df4-beb0-3dabfaa670c9.png)

Podemos observar en la siguiente imagen como quedaron los grupos
![ksnip_20220220-194641](https://user-images.githubusercontent.com/40868390/154872418-ca7ad9fc-23da-43df-b512-8208d474952a.png)

En la siguiente imagen aparece como quedron distribuidos los clusters para las dos primeras columnas del dataset
![ksnip_20220220-194902](https://user-images.githubusercontent.com/40868390/154872537-f1fac6ae-79f4-48f6-9b1c-3f7460db8f48.png)

## Predicción

8. Con el modelo entrenado podemos predecir como se agruparan nuevos datos, para ello elegimos el dataset de prueba

9. Cargar el dataset de pruebas en weka para realizar la prediccion
![ksnip_20220220-201241](https://user-images.githubusercontent.com/40868390/154873749-28530ac8-f463-4b3d-a979-ff79b38d5879.png)

10. Finalmente se observa la prediccion realizada con los nuevos datos 
![ksnip_20220220-201449](https://user-images.githubusercontent.com/40868390/154873870-829bf43a-68b5-448f-a929-71411e409ff6.png)

En la siguiente imagen aparece como quedron distribuidos los clusters para las dos primeras columnas del nuevo dataset
![ksnip_20220220-201623](https://user-images.githubusercontent.com/40868390/154873988-415e4a51-a3f5-4370-b1c8-d6a22e46184a.png)

# Conclusión
- 
# Bibliografía
- [1] Tutorials Point, “What is Weka?”. Disponible en: https://www.tutorialspoint.com/weka/what_is_weka.htm (consultado Feb. 19, 2022).
- [2] Tutorials Point, Machine Learning with Python. Disponible en: https://www.tutorialspoint.com/machine_learning_with_python/machine_learning_with_python_tutorial.pdf.
- [3] Qué son los árboles de decisión y para qué sirven | Máxima Formación. (n.d.). Retrieved February 13, 2022, from https://www.maximaformacion.es/blog-dat/que-son-los-arboles-de-decision-y-para-que-sirven/
- [4] El algoritmo J48-Graft Tree también es capaz de trabajar con distintos tipos de atributos y lidiar con los problemas típicos de ruido (Noise). (n.d.). Retrieved February 19, 2022, from https://1library.co/article/algoritmo-graft-trabajar-distintos-atributos-lidiar-problemas-t%C3%ADpicos.zx52jowq
- [5] Ihya, R., Namir, A., el Filali, S., Ait Daoud, M., & Guerss, F. Z. (2019). J48 algorithms of machine learning for predicting user’s the acceptance of an E-orientation Systems. ACM International Conference Proceeding Series. https://doi.org/10.1145/3368756.3368995 
- [6]Decision Stump - RapidMiner Documentation. (n.d.). Retrieved February 19, 2022, from https://docs.rapidminer.com/latest/studio/operators/modeling/predictive/trees/decision_stump.html
- [7] Webb, G. I., Fürnkranz, J., Fürnkranz, J., Fürnkranz, J., Hinton, G., Sammut, C., Sander, J., Vlachos, M., Teh, Y. W., Yang, Y., Mladeni, D., Brank, J., Grobelnik, M., Zhao, Y., Karypis, G., Craw, S., Puterman, M. L., & Patrick, J. (2011). Decision Stump. Encyclopedia of Machine Learning, 262–263. https://doi.org/10.1007/978-0-387-30164-8_202
- [8] Landwehr, N., Hall, M., & Frank, E. (n.d.). Logistic Model Trees
- [9] Mishra, A. K., & Ratha, B. K. (n.d.). Study of Random Tree and Random Forest Data Mining Algorithms for Microarray Data Analysis. International Journal on Advanced Electrical and Computer Engineering.
- [10] Youtube.com. 2022. [online] Valido en: <https://www.youtube.com/watch?v=EItlUEPCIzM> [Accessed 17 February 2022].
- [11] La tecnología cambia la vida futura – Gobetech.com. 2022. ¿Cuáles son las ventajas de la agrupación de K-Means?. [online] Valido en: <https://tech.gobetech.com/31740/cuales-son-las-ventajas-de-la-agrupacion-de-k-means.html> [Acceso 19 February 2022].
- [12] The Machine Learners. 2022. Algoritmo K-Means | Clustering de forma sencilla. [online] Valido en: <https://www.themachinelearners.com/k-means/> [Acceso 19 February 2022].
