#Ejercicio 2.1

import numpy as np
import matplotlib.pyplot as plt
import funciones_auxiliares as fa
import os
from PIL import Image

#importar las imagenes de la carpeta dataset_img como vectores

#1. Aprender una representación basada en Descomposición de Valores Singulares utilizando las n imágenes.

#Cargar las imágenes
img_folder = 'datasets_imgs'
img_files = os.listdir(img_folder)
img_files = [f for f in img_files if f.endswith('.jpeg')]
p = 28
X = np.zeros((len(img_files), p**2))
for i, f in enumerate(img_files):
    img_vector = fa.cargar_y_transformar_imagen(os.path.join(img_folder, f), p)
    if img_vector is not None:
        X[i] = img_vector


#graficar X
plt.figure()
for i in range(18):
    plt.subplot(3, 6, i+1)
    plt.imshow(X[i].reshape(p, p), cmap='gray')
    plt.axis('off')
plt.suptitle('Imágenes originales')
plt.show()

#matriz de similaridad de x
mat_simil_x = fa.matriz_de_similitud(X,1000)

#2. Calcular la matriz de similitud W de las imágenes utilizando la función matriz_de_similitud de funciones_auxiliares.py.

#3 bajar la dimensionalidad a X
k = 5
#calculemos el nuevo X_k
X_k = fa.compresion(X,k)
#graficar
plt.figure()
for i in range(18):
    plt.subplot(3, 6, i+1)
    plt.imshow(X_k[i].reshape(p, p), cmap='gray')
    plt.axis('off')
plt.suptitle('Imágenes reconstruidas')
plt.show()

#matriz de similaridad de X_k
mat_simil_X_k = fa.matriz_de_similitud(X_k,1000)

#Graficar

# plt.figure()
# plt.imshow(mat_simil_x, cmap='viridis')
# plt.colorbar()
# plt.title('Matriz de Similitud')
# plt.xlabel('Índice')
# plt.ylabel('Índice')
# plt.show()

# plt.figure()
# plt.imshow(mat_simil_X_k, cmap='viridis')
# plt.colorbar()
# plt.title('Matriz de Similitud')
# plt.xlabel('Índice')
# plt.ylabel('Índice')
# plt.show()

# #graficar esta diferencia para distintos valores de k
# k_values = np.arange(1, 30)
# diferencias = np.zeros(k_values.shape)
# for i, k in enumerate(k_values):
#     X_k = fa.compresion(X, k)
#     diferencias[i] = np.linalg.norm(X-X_k)
# plt.figure()
# plt.plot(k_values, diferencias, color='hotpink')
# plt.xlabel('k')
# plt.ylabel('Diferencia')
# plt.title('Diferencia entre las imágenes originales y las reconstruidas')
# plt.grid()
# plt.show()


# #calcular el rango de x
# rango = np.linalg.matrix_rank(X)
# # print(f'El rango de la matriz X es: {rango}')
