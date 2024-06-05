#Ejercicio 2.4

import numpy as np
import matplotlib.pyplot as plt
import os
import funciones_auxiliares as fa

"""Dado el dataset dataset_imagenes2.zip encontrar d, el número mínimo de dimensiones a las que se
puede reducir la dimensionalidad de su representación mediante valores singulares tal que el error de
cada imagen comprimida y su original no exceda el 10% bajo la norma de Frobenius. Utilizando esta
ultima representación aprendida con el dataset 2 ¿Qué error de reconstrucción obtienen si utilizan la
misma compresión (con la misma base de d dimensiones obtenida del dataset 2) para las imagenes
dataset_imagenes1.zip?"""

#Cargar las imágenes
img_folder = 'datasets_imgs_02'
img_files = os.listdir(img_folder)
img_files = [f for f in img_files if f.endswith('.jpeg')]
p = 28
X_dataset2 = np.zeros((len(img_files), p**2))
for i, f in enumerate(img_files):
    img_vector = fa.cargar_y_transformar_imagen(os.path.join(img_folder, f), p)
    if img_vector is not None:
        X_dataset2[i] = img_vector

# #graficar X
# plt.figure()
# for i in range(8):
#     plt.subplot(2, 4, i+1)
#     plt.imshow(X_dataset2[i].reshape(p, p), cmap='gray')
#     plt.axis('off')
# # plt.suptitle('Imágenes originales')
# plt.show()

#Graficar los valores singulares de X
U, S, Vt = np.linalg.svd(X_dataset2, full_matrices=False)
# plt.figure()
# plt.bar(range(1, 9), S[:8])
# plt.title('Valores singulares de X')
# plt.show()

def calcular_error_por_imagen():
    error_por_imagen = []
    for i in range(1, 9):
        imagen = []
        for j in range(1, 9):
            # Reducir a d dimensiones
            S_d = np.zeros((U.shape[0], Vt.shape[0]))
            np.fill_diagonal(S_d[:i, :i], S[:i])
            X_d = np.dot(U, np.dot(S_d, Vt))

            # Calcular el error de reconstrucción para cada imagen
            errores = np.linalg.norm(X_dataset2 - X_d, axis=1) / np.linalg.norm(X_dataset2, axis=1)
            imagen.append(errores[j-1])
        error_por_imagen.append(imagen)
    return error_por_imagen

# Crear una nueva figura
plt.figure()

error_por_imagen = calcular_error_por_imagen()

# Generar una función para cada imagen
for i in range(8):
    # Extraer los errores de la imagen i
    errores_imagen_i = [error_por_imagen[j][i] for j in range(8)]
    
    # Graficar los errores de la imagen i
    plt.plot(range(1, 9), errores_imagen_i, label=f'Imagen {i+1}')

plt.axhline(y=0.1, color='r', linestyle='--')


# Añadir una leyenda
plt.legend()

# Añadir títulos a los ejes y a la figura
plt.xlabel('Dimensiones')
plt.ylabel('Error de reconstrucción')
# plt.title('Error de reconstrucción por número de dimensiones')

# Mostrar el gráfico
plt.show()

#calculo para imagen la dimensión a partir de la cual el error es menor o igual a 0.1
dimensiones_minimas_por_imagen = []
for i in range(len(error_por_imagen)):
    for j in range(len(error_por_imagen[i])):
        if error_por_imagen[j][i] <= 0.1:  # Corrección aquí
            dimensiones_minimas_por_imagen.append(j+1)
            break

print(dimensiones_minimas_por_imagen)

# debido a que debemos encontrar la menor dimension que haga que los errores de todas las imagenes sea menor al 10%, tomamos el maximo de las dimensiones minimas por imagen    
dimension_minima = max(dimensiones_minimas_por_imagen)   

#nos quedo d = 8

#Segunda parte del ejercicio
"""Utilizando esta ultima representación aprendida con el dataset 2 ¿Qué error de reconstrucción 
obtienen si utilizan la misma compresión (con la misma base de d dimensiones obtenida del dataset 2)
para las imagenes dataset_imagenes1.zip?"""

#funcion para obtener la matriz utilizada para reducir dimensión en PCA
#obtenemos el V_5 utilizado para la compresión con SVD del dataset_2
dimension_maxima = max(dimensiones_minimas_por_imagen)
# Vt_8 = fa.obtener_matriz_de_proyeccion(X_dataset2, dimension_maxima)

#Cargar las imágenes del dataset 1
from Ejercicio2_2 import X as X_dataset1
# Vt_8_red = Vt_8[:8,:]
# print(Vt_8_red.shape)
#Reducir la dimensionalidad de las imágenes del dataset 1 con V_5
U,S,Vt = np.linalg.svd(X_dataset2)
Vt_8 = Vt[:8,:]
X_dataset1_reducido = X_dataset1 @ (Vt_8.T @ Vt_8)

#Calcular el error de reconstrucción CON la norma de FROBENIUS
error_de_reconstruccion_dataset1 = np.linalg.norm(X_dataset1 - X_dataset1_reducido) / np.linalg.norm(X_dataset1)

print(f"El error de reconstrucción para las imágenes del dataset 1 utilizando la compresión con d=8 del dataset 2 es: {error_de_reconstruccion_dataset1}")

#graficamos como quedan los datos del dataset 1 (X_dataset1_reducido)
fig, axs = plt.subplots(1, 19, figsize = (19,2))
for i in range(19):
    vector = X_dataset1_reducido[i].reshape(28,28)
    axs[i].imshow(vector, cmap='gray')
    axs[i].axis('off')
plt.show()