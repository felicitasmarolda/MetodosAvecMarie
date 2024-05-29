#trabajo práctico 3 de MNyO

import numpy as np
import matplotlib.pyplot as plt
import funciones_auxiliares as fa

#Cargar el dataset X desde el archivo dataset.csv. en la carpeta dataset
X = np.loadtxt('dataset.csv', delimiter=',', skiprows=1)
X = X[:, 1:]

#HACEMOS PCA
#TODO descomentar
#calculamos los componentes principales de X
m, n = X.shape
Z = fa.PCA(X, n)

# calculamos los componentes principales de X_2
Z_2 = fa.PCA(X, 2)

#TODO descomentar
# #Graficamos los componentes principales de X_2
plt.figure()
plt.scatter(Z_2[:, 0], Z_2[:, 1])
plt.title('Componentes principales de X_2')
plt.show()

Z_6 = fa.PCA(X, 6)
Z_10 = fa.PCA(X, 10)

# #hacer las matrices de similaridad del PCA
# #TODO descomentar
W_PCA = fa.matriz_de_similitud(Z, 10)
W_PCA_2 = fa.matriz_de_similitud(Z_2, 10)
W_PCA_6 = fa.matriz_de_similitud(Z_6, 10)
W_PCA_10 = fa.matriz_de_similitud(Z_10, 10)


# Crear una figura y una cuadrícula de subplots 2x2
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Graficar la matriz de similitud de Z en el primer subplot
im1 = axs[0, 0].imshow(W_PCA, cmap='viridis')
axs[0, 0].set_title('Matriz de similitud de Z')

# Graficar la matriz de similitud de Z_2 en el segundo subplot
im2 = axs[1, 1].imshow(W_PCA_2, cmap='viridis')
axs[1, 1].set_title('Matriz de similitud de Z_2')

# Graficar la matriz de similitud de Z_6 en el tercer subplot
im3 = axs[1, 0].imshow(W_PCA_6, cmap='viridis')
axs[1, 0].set_title('Matriz de similitud de Z_6')

# Graficar la matriz de similitud de Z_10 en el cuarto subplot
im4 = axs[0, 1].imshow(W_PCA_10, cmap='viridis')
axs[0, 1].set_title('Matriz de similitud de Z_10')

# Ajustar el espacio para la barra de colores
plt.subplots_adjust(right=0.8)

# Crear una barra de colores a la derecha de todos los subplots
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im1, cax=cbar_ax)

# Mostrar la figura
plt.show()


#graficar los valores singulares de X en grafico de barras
plt.figure()
plt.bar(range(1, len(S) + 1), S)
plt.title('Valores singulares de X')
plt.xlabel('Número de valor singular')
plt.ylabel('Valor singular')
plt.show()


#FALTA
 # probar porque elegimos el valor de sigma