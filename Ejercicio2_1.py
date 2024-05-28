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


# #graficar X
# plt.figure()
# for i in range(10):
#     plt.subplot(2, 5, i+1)
#     plt.imshow(X[i].reshape(p, p), cmap='gray')
#     plt.axis('off')
# plt.suptitle('Imágenes originales')
# plt.show()


# #2. Calcular la matriz de similitud W de las imágenes utilizando la función matriz_de_similitud de funciones_auxiliares.py.

# #hacer svd
# U, S, Vt = np.linalg.svd(X, full_matrices=False)

# #3 bajar la dimensionalidad a X
# k = 18
# U_k, S_k, Vt_k = fa.calcular_X_k(U, S, Vt, k)
# #calculemos el nuevo X_k
# X_k = np.dot(U_k, np.dot(np.diag(S_k), Vt_k))
# #graficar
# plt.figure()
# for i in range(10):
#     plt.subplot(2, 5, i+1)
#     plt.imshow(X_k[i].reshape(p, p), cmap='gray')
#     plt.axis('off')
# plt.suptitle('Imágenes reconstruidas')
# plt.show()

# #hacer una formula que calcule la diferencia entre las imagenes originales y las reconstruidas
# diferencia = np.linalg.norm(X-X_k)
# # print(f'La diferencia entre las imágenes originales y las reconstruidas es: {diferencia}')

# #graficar esta diferencia para distintos valores de k
# k_values = np.arange(1, 30)
# diferencias = np.zeros(k_values.shape)
# for i, k in enumerate(k_values):
#     U_k, S_k, Vt_k = fa.calcular_X_k(U, S, Vt, k)
#     X_k = np.dot(U_k, np.dot(np.diag(S_k), Vt_k))
#     diferencias[i] = np.linalg.norm(X-X_k)
# plt.figure()
# plt.plot(k_values, diferencias)
# plt.xlabel('k')
# plt.ylabel('Diferencia')
# plt.title('Diferencia entre las imágenes originales y las reconstruidas')
# plt.grid()
# plt.show()

# #calcular el rango de x
# rango = np.linalg.matrix_rank(X)
# print(f'El rango de la matriz X es: {rango}')