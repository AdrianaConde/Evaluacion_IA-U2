# **Árbol de decisión**

Un Árbol de Decisión es un método analítico que a través de una representación esquemática de las alternativas disponibles facilita la toma de mejores decisiones, especialmente cuando existen riesgos, costos, beneficios y múltiples opciones. El nombre se deriva de la apariencia del modelo parecido a un árbol y su uso es amplio en el ámbito de la toma de decisiones bajo incertidumbre.  

Estructura básica de un árbol de decisión
Los árboles de decisión están formados por nodos y su lectura se realiza de arriba hacia abajo. 

Dentro de un árbol de decisión distinguimos diferentes tipos de nodos: 

* Primer nodo o nodo raíz: en él se produce la primera división en función de la variable más importante.
* Nodos internos o intermedios: tras la primera división encontramos estos nodos, que vuelven a dividir el conjunto de datos en función de las variables.
* Nodos terminales u hojas: se ubican en la parte inferior del esquema y su función es indicar la clasificación definitiva.

![Image text](https://www.maximaformacion.es/wp-content/uploads/2021/07/Estructura-de-un-arbol-de-decision.jpg)

#### *Ventajas*

* Son fáciles de construir, interpretar y visualizar.
* Selecciona las variables más gimportantes y en su creación no siempre se hace uso de todos los predictores.
* Si faltan datos no podremos recorrer el árbol hasta un nodo terminal, pero sí podemos hacer predicciones promediando las hojas del sub-árbol que alcancemos.
* No es preciso que se cumplan una serie de supuestos como en la regresión lineal (linealidad, normalidad de los residuos, homogeneidad de la varianza, etc.).
* Sirven tanto para variables dependientes cualitativas como cuantitativas, como para variables predictoras o independientes numéricas y categóricas. Además, no necesita variables dummys, aunque a veces mejoran el modelo.
* Permiten relaciones no lineales entre las variables explicativas y la variable dependiente.
* Nos podemos servir de ellos para categorizar variables numéricas.

#### *Desventajas*

* Tienden al sobreajuste u overfitting de los datos, por lo que el modelo al predecir nuevos casos no estima con el mismo índice de acierto.
* Se ven influenciadas por los outliers, creando árboles con ramas muy profundas que no predicen bien para nuevos casos. Se deben eliminar dichos outliers.
* No suelen ser muy eficientes con modelos de regresión.
* Crear árboles demasiado complejos puede conllevar que no se adapten bien a los nuevos datos. La complejidad resta capacidad de interpretación.
* Se pueden crear árboles sesgados si una de las clases es más numerosa que otra.
* Se pierde información cuando se utilizan para categorizar una variable numérica continua.

# **K-means** 

El algoritmo de agrupación en clústeres de K-Means se define como un método de aprendizaje no supervisado que tiene un proceso iterativo en el que el conjunto de datos se agrupa en un número k de clústeres o subgrupos predefinidos que no se superponen, haciendo que los puntos internos del clúster sean lo más similares posible mientras se intenta mantener el agrupa en un espacio distinto,

Se utiliza cuando tenemos un montón de datos sin etiquetar. El objetivo de este algoritmo es el de encontrar “K” grupos (clusters) entre los datos crudos. 

Los “centroides” de cada grupo que serán unas “coordenadas” de cada uno de los K conjuntos que se utilizarán para poder etiquetar nuevas muestras.
Etiquetas para el conjunto de datos de entrenamiento. Cada etiqueta perteneciente a uno de los K grupos formados.

#### *Algoritmo*
1. Los puntos K se colocan en el espacio de datos del objeto que representa el grupo inicial de centroides.
2. Cada objeto o punto de datos se asigna a la k más cercana.
3. Una vez asignados todos los objetos, se vuelven a calcular las posiciones de los k centroides.
4. Los pasos 2 y 3 se repiten hasta que las posiciones de los centroides dejen de moverse




