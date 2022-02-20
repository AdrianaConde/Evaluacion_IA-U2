import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Variables
k = 3 # -> numero de clusters(agrupamientos)
path_csv = ''
name_entrenamiento = 'entreamiento.csv'
name_prueba = 'test.csv'
dataset_entrenamiento = name_entrenamiento
dataset_prueba = name_prueba

# Asignacion del dataset a la variable datos
datos = pd.read_csv(dataset_entrenamiento)
datosp = pd.read_csv(dataset_prueba)

print("-------------------------")
print('Dimensiones del archivo de datos -> ', name_entrenamiento,':\n', datos.shape)
print("-------------------------")
print("-------------------------")
print('Dimensiones del archivo de pruebas -> ', name_prueba,':\n', datosp.shape)
print("-------------------------")

# Se extrae los valores del dataset de entrenamiento y pruebas a un array
x = datos.iloc[:,1:26].values
y = datosp.iloc[:,1:26].values

# Normalizacion de los datos
datos_norm = (x-x.min())/(x.max()-x.min())
datos_norm1 = (y-y.min())/(y.max()-y.min())

# Aqui se crean solo dos columnas que representan todas las columnas del dataset
pca2 = PCA(n_components=2)
pca_datos2 = pca2.fit_transform(datos_norm)
pca_datos_df2 = pd.DataFrame(data = pca_datos2, columns = ['A1', 'A2'])

#crea la figura con los puntos en dos dimenciones de los datos originales
fig, plt3 = plt.subplots(figsize=(10,5))
plt3.scatter(pca_datos_df2.A1, pca_datos_df2.A2, s=10, c='b')
plt3.set_title('Datos originales')

""" -------------------------------metodo del codo para saber el numero de centroides a usar----------------------------------------------"""
wcss=[]
for i in range(1,11):
	kmeans = KMeans(i)
	kmeans.fit(datos_norm)
	wcss_iter = kmeans.inertia_
	wcss.append(wcss_iter)
number_clusters = range(1,11)

fig, plt1 = plt.subplots(figsize=(10,5))
plt1.plot(number_clusters, wcss)

plt1.set_title('Metodo del codo')
plt1.set_xlabel('numero de clusters')
plt1.set_ylabel('WCSS')

print('----------------------------------')
print("Agoritmo KMeans")
print('----------------------------------')

""" ---------------------------------------------metodo kmeans-----------------------------------------------------"""
km = KMeans(n_clusters = k, max_iter = 300, random_state = 0)
km.fit(datos_norm)

iteraciones = km.n_iter_
print('------------------------')
print("Nro de iteraciones: ", iteraciones)
print('------------------------')

datos_en_cluster1 = pd.Series(km.labels_).value_counts()
print('----------------------------------')
print("Nro de datos en cada grupo:\n", datos_en_cluster1)
print('----------------------------------')

datos['KMeans_Clusters'] = km.labels_

# Aqui se crean solo dos columnas que representan todas las columnas del dataset
pca = PCA(n_components=2)
pca_datos = pca.fit_transform(datos_norm)
pca_datos_df = pd.DataFrame(data = pca_datos, columns = ['Componente1', 'Componente2'])
pca_nombre_datos = pd.concat([pca_datos_df, datos[['KMeans_Clusters']]], axis=1)

pca1 = PCA(n_components=2)
pca_datos1 = pca1.fit_transform(km.cluster_centers_)
pca_datos_df1 = pd.DataFrame(data = pca_datos1, columns = ['C1', 'C2'])
print("Centroides:\n", pca_datos_df1)

#crea la figura con los puntos en dos dimenciones de los datos de entrenamiento clasificados
fig, plt2 = plt.subplots(figsize=(10,5))
for i in range(k):
	colors = np.array(['blue', 'green', 'yellow', 'orange', 'brown', 'pink', 'purple', 'black', 'gray'])
	plt2.scatter(x = pca_nombre_datos.Componente1, y = pca_nombre_datos.Componente2, s=10, c=colors[pca_nombre_datos.KMeans_Clusters])
plt2.scatter(x = pca_datos_df1.C1, y = pca_datos_df1.C2, s=30, c='red')
plt2.set_title('Algoritmo k-means')

"""---------------------------------Prediccion de datos usando el modelo entrenado-------------------------------------------"""
d = km.predict(datos_norm1)
print(d)

datosp['KMeans_Clusters'] = d

# Aqui se crean solo dos columnas que representan todas las columnas del dataset
pca3 = PCA(n_components=2)
pca_datos3 = pca3.fit_transform(datos_norm1)
pca_datos_df3 = pd.DataFrame(data = pca_datos3, columns = ['B1', 'B2'])
pca_nombre_datos3 = pd.concat([pca_datos_df3, datosp[['KMeans_Clusters']]], axis=1)

pca4 = PCA(n_components=2)
pca_datos4 = pca4.fit_transform(km.cluster_centers_)
pca_datos_df4 = pd.DataFrame(data = pca_datos4, columns = ['D1', 'D2'])
print("Centroides:\n", pca_datos_df4)

#crea la figura con los puntos en dos dimenciones de los datos de prueba clasificados
fig, plt4 = plt.subplots(figsize=(10,5))
for i in range(k):
	colors = np.array(['blue', 'green', 'yellow', 'orange', 'brown', 'pink', 'purple', 'black', 'gray'])
	plt4.scatter(x = pca_nombre_datos3.B1, y = pca_nombre_datos3.B2, s=10, c=colors[pca_nombre_datos3.KMeans_Clusters])
plt4.scatter(x = pca_datos_df4.D1, y = pca_datos_df4.D2, s=30, c='red')
plt4.set_title('Predicción de algoritmo k-means')

plt.show()
