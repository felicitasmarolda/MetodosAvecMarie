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
from Ejercicio2_1 import X_k


#matriz_similaridad = fa.similaridad_entre_pares_de_imagenes(X)
#graficar la matriz de similaridad
# plt.figure()
# plt.imshow(matriz_similaridad[:, :, 0], cmap='hot')
# plt.colorbar()
# plt.title('Similaridad entre pares de imágenes')
# plt.xticks(np.arange(0, 19, step=1))
# plt.yticks(np.arange(0, 19, step=1))
# plt.show()# Crear una figura


fig, axs = plt.subplots(3, 6, figsize=(15, 10))

# Iterar sobre los valores de k de 2 a 19
for k in range(2, 20):
    matriz_similaridad = fa.similaridad_entre_pares_de_imagenes(X)[:, :, k]

    # Graficar la matriz de similaridad en un subplot
    ax = axs[(k-2)//6, (k-2)%6]
    im = ax.imshow(matriz_similaridad, cmap='hot')
    ax.set_title(f'Similaridad para k={k}')

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax)


# Mostrar la figura
plt.show()