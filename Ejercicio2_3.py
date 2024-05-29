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
from Ejercicio2_1 import X

#calculamos 3 X_k
X_10 = fa.compresion(X,10)
X_5 = fa.compresion(X,5)
X_1 = fa.compresion(X,1)

sigma = 500
max_simil_X = fa.matriz_de_similitud(X,sigma)
mat_simil_10 = fa.matriz_de_similitud(X_10,sigma) 
mat_simil_5 = fa.matriz_de_similitud(X_5,sigma) 
mat_simil_1 = fa.matriz_de_similitud(X_1,sigma) 

fig, axes = plt.subplots(1, 4, figsize=(20, 6))
im1 = axes[0].imshow(mat_simil_10, cmap='viridis')
axes[0].set_title('Matriz de Similitud 10')
axes[0].set_xlabel('Índice')
axes[0].set_ylabel('Índice')
fig.colorbar(im1, ax=axes[0])

# Graficar mat_simil_5
im2 = axes[1].imshow(mat_simil_5, cmap='viridis')
axes[1].set_title('Matriz de Similitud 5')
axes[1].set_xlabel('Índice')
axes[1].set_ylabel('Índice')
fig.colorbar(im2, ax=axes[1])

# Graficar mat_simil_1
im3 = axes[2].imshow(mat_simil_1, cmap='viridis')
axes[2].set_title('Matriz de Similitud 1')
axes[2].set_xlabel('Índice')
axes[2].set_ylabel('Índice')
fig.colorbar(im3, ax=axes[2])

im4 = axes[3].imshow(max_simil_X,cmap='viridis')
axes[3].set_title('Matriz de Similitud X')
axes[3].set_xlabel('Índice')
axes[3].set_ylabel('Índice')
fig.colorbar(im4, ax=axes[3])

plt.tight_layout()
plt.show()