import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

k = 3 # -> numero de clusters(agrupamientos)
name = 'dataset.csv'
path_csv = '../Dataset/'
dataset = path_csv + name
datos = pd.read_csv(dataset)

print("-------------------------")
print('Dimensiones del archivo de datos', name,':\n', datos.shape)
print("-------------------------")

x = datos.iloc[:,1:26].values

datos_norm = (x-x.min())/(x.max()-x.min())

# metodo del codo para saber el numero de centroides a usar
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

# metodo kmeans 
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

pca = PCA(n_components=2)
pca_datos = pca.fit_transform(datos_norm)
pca_datos_df = pd.DataFrame(data = pca_datos, columns = ['Componente1', 'Componente2'])
pca_nombre_datos = pd.concat([pca_datos_df, datos[['KMeans_Clusters']]], axis=1)

pca1 = PCA(n_components=2)
pca_datos1 = pca1.fit_transform(km.cluster_centers_)
pca_datos_df1 = pd.DataFrame(data = pca_datos1, columns = ['C1', 'C2'])
print("Centroides:\n", pca_datos_df1)

fig, plt2 = plt.subplots(figsize=(10,5))
for i in range(k):
	colors = np.array(['blue', 'green', 'yellow', 'orange', 'brown', 'pink', 'purple', 'black', 'gray'])
	plt2.scatter(x = pca_nombre_datos.Componente1, y = pca_nombre_datos.Componente2, s=10, c=colors[pca_nombre_datos.KMeans_Clusters])
plt2.scatter(x = pca_datos_df1.C1, y = pca_datos_df1.C2, s=30, c='red')
plt2.set_title('Algoritmo k-means')
plt.show()

datos.to_csv('dataset_clustering.csv')
