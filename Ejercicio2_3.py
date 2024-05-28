#Ejercicio 2.3

import numpy as np
import matplotlib.pyplot as plt
import funciones_auxiliares as fa
import Ejercicio2_1 as ej2_1


"""Utilizando compresión con distintos valores de d medir la similaridad entre pares de imágenes (con
alguna métrica de similaridad que decida el autor) en un espacio de baja dimensión d. Analizar cómo
la similaridad entre pares de imágenes cambia a medida que se utilizan distintos valores de d. Cuales
imágenes se encuentran cerca entre si? Alguna interpretación al respecto? Ayuda: ver de utilizar una
matriz de similaridad para visualizar todas las similaridades par-a-par juntas."""

#importamos la matriz de imagenes del ejercicio 2_1
from Ejercicio2_1 import X

matriz_similaridad = fa.similaridad_entre_pares_de_imagenes(X)
#graficar la matriz de similaridad
plt.figure()
plt.imshow(matriz_similaridad[:, :, 0], cmap='hot')
plt.colorbar()
plt.title('Similaridad entre pares de imágenes')
plt.xticks(np.arange(0, 19, step=1))
plt.yticks(np.arange(0, 19, step=1))
plt.show()
