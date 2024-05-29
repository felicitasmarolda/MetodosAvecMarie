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
X = np.zeros((len(img_files), p**2))
for i, f in enumerate(img_files):
    img_vector = fa.cargar_y_transformar_imagen(os.path.join(img_folder, f), p)
    if img_vector is not None:
        X[i] = img_vector

# #graficar X
# plt.figure()
# for i in range(8):
#     plt.subplot(2, 4, i+1)
#     plt.imshow(X[i].reshape(p, p), cmap='gray')
#     plt.axis('off')
# plt.suptitle('Imágenes originales')
# plt.show()

#Graficar los valores singulares de X
U, S, Vt = np.linalg.svd(X, full_matrices=False)
plt.figure()
plt.bar(range(1, 9), S[:8])
plt.title('Valores singulares de X')
plt.show()


# def encontrar_dimensiones_minimas(X, max_error=0.1):
#     # Realizar la descomposición en valores singulares de X
#     U, S, Vt = np.linalg.svd(X, full_matrices=False)

#     # Calcular la norma de Frobenius de X
#     norma_frobenius_X = np.linalg.norm(X, 'fro')

#     for d in range(1, X.shape[1] + 1):
#         # Reducir a d dimensiones
#         S_d = np.zeros((U.shape[0], Vt.shape[0]))
#         np.fill_diagonal(S_d[:d, :d], S[:d])
#         X_d = np.dot(U, np.dot(S_d, Vt))

#         # Calcular el error de reconstrucción para cada imagen
#         errores = np.sqrt(np.sum((X - X_d)**2, axis=1))

#         # Si el error máximo no excede el 10% de la norma de Frobenius de X, devolver d
#         if np.max(errores) <= max_error * norma_frobenius_X:
#             return d

#     # Si no se encontró un d que cumpla con el criterio, devolver None

#     return None

# d = encontrar_dimensiones_minimas(X)
# print(f'El número mínimo de dimensiones es {d}')



error_por_imagen = []
for i in range(1, 9):
    imagen = []
    for j in range(1, 9):
        # Reducir a d dimensiones
        S_d = np.zeros((U.shape[0], Vt.shape[0]))
        np.fill_diagonal(S_d[:i, :i], S[:i])
        X_d = np.dot(U, np.dot(S_d, Vt))

        # Calcular el error de reconstrucción para cada imagen
        errores = np.linalg.norm(X - X_d, axis=1) / np.linalg.norm(X, axis=1)
        imagen.append(errores[j-1])
    error_por_imagen.append(imagen)

print(error_por_imagen)


# Crear una nueva figura
plt.figure()

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
plt.title('Error de reconstrucción por número de dimensiones')

# Mostrar el gráfico
plt.show()
