#Ejercicio 2.2

import numpy as np
import matplotlib.pyplot as plt
import funciones_auxiliares as fa
import os
from PIL import Image

# Defino la carpeta donde están las imágenes
img_folder = 'datasets_imgs'

# Obtengo la lista de archivos en la carpeta
img_files = os.listdir(img_folder)

# Filtro la lista para quedarme solo con los archivos .jpeg
img_files = [f for f in img_files if f.endswith('.jpeg')]

# Defino el tamaño de las imágenes
p = 28

# Inicializo la matriz X donde voy a guardar las imágenes
X = np.zeros((len(img_files), p**2))

# Cargo las imágenes en la matriz X
for i, f in enumerate(img_files):
    img_vector = fa.cargar_y_transformar_imagen(os.path.join(img_folder, f), p)
    if img_vector is not None:
        X[i] = img_vector

# Muestro las primeras 18 imágenes
plt.figure()
for i in range(18):
    plt.subplot(3, 6, i+1)
    plt.imshow(X[i].reshape(p, p), cmap='gray')
    plt.axis('off')
plt.show()

# Calculo la matriz de similitud de X
mat_simil_x = fa.matriz_de_similitud(X,1000)

# Comprimo las imágenes a una dimensión y las muestro
k = 1
X_k = fa.compresion(X,k)
plt.figure()
for i in range(18):
    plt.subplot(3, 6, i+1)
    plt.imshow(X_k[i].reshape(p, p), cmap='gray')
    plt.axis('off')
plt.show()

# Calculo la matriz de similitud de las imágenes comprimidas
mat_simil_X_k = fa.matriz_de_similitud(X_k,1000)

# Comprimo las imágenes a diferentes dimensiones
X_1 = fa.compresion(X,1)
X_3 = fa.compresion(X,3)
X_5 = fa.compresion(X,5)
X_8 = fa.compresion(X,8)
X_10 = fa.compresion(X,10)
X_15 = fa.compresion(X,15)

# Muestro las imágenes comprimidas a diferentes dimensiones
plt.figure(figsize=(18, 6))
for n in range(1, 7):
    for i in range(18):
        plt.subplot(6, 18, (n-1)*18+i+1) 
        plt.axis('off')
plt.show()

# Muestro la matriz de similitud de las imágenes comprimidas
plt.figure()
plt.imshow(mat_simil_X_k, cmap='viridis')
plt.colorbar()
plt.title('Matriz de Similitud')
plt.xlabel('Índice')
plt.ylabel('Índice')
plt.show()

# Calculo la diferencia entre las imágenes originales y las comprimidas para diferentes valores de k
k_values = np.arange(1, 30)
diferencias = np.zeros(k_values.shape)
for i, k in enumerate(k_values):
    X_k = fa.compresion(X, k)
    diferencias[i] = np.linalg.norm(X-X_k)

# Muestro la diferencia en función de k
plt.figure()
plt.plot(k_values, diferencias, color='hotpink')
plt.xlabel('k')
plt.ylabel('Diferencia')
plt.title('Diferencia entre las imágenes originales y las reconstruidas')
plt.grid()
plt.show()

# Imprimo el rango de la matriz X
rango = np.linalg.matrix_rank(X)
print(f'El rango de la matriz X es: {rango}')