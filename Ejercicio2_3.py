#Ejercicio 2.3

import numpy as np
import matplotlib.pyplot as plt
import funciones_auxiliares as fa


"""Utilizando compresión con distintos valores de d medir la similaridad entre pares de imágenes (con
alguna métrica de similaridad que decida el autor) en un espacio de baja dimensión d. Analizar cómo
la similaridad entre pares de imágenes cambia a medida que se utilizan distintos valores de d. Cuales
imágenes se encuentran cerca entre si? Alguna interpretación al respecto? Ayuda: ver de utilizar una
matriz de similaridad para visualizar todas las similaridades par-a-par juntas."""

#importamos la matriz de imagenes del ejercicio 2_1
from Ejercicio2_2 import X

#calculamos 3 X_k
X_15 = fa.compresion(X,15)
X_10 = fa.compresion(X,10)
X_5 = fa.compresion(X,5)
X_1 = fa.compresion(X,1)

sigma = 800
max_simil_X = fa.matriz_de_similitud(X,sigma)
mat_simil_15 = fa.matriz_de_similitud(X_15,sigma)
mat_simil_10 = fa.matriz_de_similitud(X_10,sigma) 
mat_simil_5 = fa.matriz_de_similitud(X_5,sigma) 
mat_simil_1 = fa.matriz_de_similitud(X_1,sigma) 

# Primera figura
fig1, ax1 = plt.subplots(figsize=(6, 4))
im3 = ax1.imshow(mat_simil_1, cmap='viridis')
# ax1.set_title('Matriz de Similitud con dimensión 1')
fig1.colorbar(im3, ax=ax1)
plt.show()

# # Segunda figura
# fig2, axes = plt.subplots(1, 4, figsize=(26, 6))  # Cambiado de (1, 3) a (1, 4)

# # Cambiando el orden de las gráficas a 5, 10, 15, original
# im2 = axes[0].imshow(mat_simil_5, cmap='viridis')
# axes[0].set_title('Dimensión 5')
# fig2.colorbar(im2, ax=axes[0])

# im1 = axes[1].imshow(mat_simil_10, cmap='viridis')
# axes[1].set_title('Dimensión 10')
# fig2.colorbar(im1, ax=axes[1])

# im5 = axes[2].imshow(mat_simil_15, cmap='viridis')
# axes[2].set_title('Dimensión 15')
# fig2.colorbar(im5, ax=axes[2])

# im4 = axes[3].imshow(max_simil_X,cmap='viridis')
# axes[3].set_title('X original')
# fig2.colorbar(im4, ax=axes[3])

# # Asegurándose de que los ejes x e y muestren los mismos números
# # Cambiando el paso de los ticks a 5
# for ax in axes:
#     ax.set_xticks(range(0, len(mat_simil_10), 5))
#     ax.set_yticks(range(0, len(mat_simil_10), 5))

# plt.tight_layout(pad=5.0)  # Agregado pad=3.0 para agregar espacio entre las subtramas
# plt.show()

# # Grafica las imágenes 03 y 15 original y las compresiones en dimensión 5 en un mismo gráfico
# plt.figure()

# plt.subplot(2, 2, 1)  # Cambiado de (1, 4, 1) a (2, 2, 1)
# plt.imshow(X[3].reshape(28, 28), cmap='gray')
# plt.title('Imagen 3 original')
# plt.axis('off')

# plt.subplot(2, 2, 2)  # Cambiado de (1, 4, 2) a (2, 2, 2)
# plt.imshow(X_5[3].reshape(28, 28), cmap='gray')
# plt.title('Imagen 3 con dimensión 5')
# plt.axis('off')

# plt.subplot(2, 2, 3)  # Cambiado de (1, 4, 3) a (2, 2, 3)
# plt.imshow(X[15].reshape(28, 28), cmap='gray')
# plt.title('Imagen 15 original')
# plt.axis('off')

# plt.subplot(2, 2, 4)  # Cambiado de (1, 4, 4) a (2, 2, 4)
# plt.imshow(X_5[15].reshape(28, 28), cmap='gray')
# plt.title('Imagen 15 con dimensión 5')
# plt.axis('off')

# plt.tight_layout()  # Asegura que las subtramas no se superpongan
# plt.show()




