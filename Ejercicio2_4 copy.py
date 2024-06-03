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
#     plt.imshow(X[i].reshape(p, p), cmap='gray')
#     plt.axis('off')
# plt.suptitle('Imágenes originales')
# plt.show()

#Graficar los valores singulares de X
U, S, Vt = np.linalg.svd(X_dataset2, full_matrices=False)
plt.figure()
plt.bar(range(1, 9), S[:8])
plt.title('Valores singulares de X')
plt.show()
# Calcular la norma de Frobenius de la matriz original
fro_norm_original = np.linalg.norm(X_dataset2, 'fro')

# Encontrar el número mínimo de dimensiones d
fro_norms = []
for d in range(1, len(S) + 1):
    S_d = np.diag(S[:d])
    U_d = U[:, :d]
    Vt_d = Vt[:d, :]
    X_d = np.dot(U_d, np.dot(S_d, Vt_d))
    fro_norm_d = np.linalg.norm(X_dataset2 - X_d, 'fro')
    fro_norms.append(fro_norm_d)
    if fro_norm_d / fro_norm_original <= 0.1:
        d_min = d
        break

print(f"El número mínimo de dimensiones d es: {d_min}")

# Graficar el error de reconstrucción
plt.figure()
plt.plot(range(1, len(S) + 1), [fn / fro_norm_original for fn in fro_norms], marker='o')
plt.axhline(y=0.1, color='r', linestyle='--')
plt.xlabel('Número de dimensiones d')
plt.ylabel('Error de reconstrucción relativo')
plt.title('Error de reconstrucción en función del número de dimensiones')
plt.show()

# Graficar los errores para las diferentes dimensiones en cada imagen
errors_per_image = []
for img_idx in range(X_dataset2.shape[0]):
    errors_img = []
    for d in range(1, len(S) + 1):
        S_d = np.diag(S[:d])
        U_d = U[:, :d]
        Vt_d = Vt[:d, :]
        X_d = np.dot(U_d, np.dot(S_d, Vt_d))
        fro_norm_d_img = np.linalg.norm(X_dataset2[img_idx] - X_d[img_idx])
        errors_img.append(fro_norm_d_img / np.linalg.norm(X_dataset2[img_idx]))
    errors_per_image.append(errors_img)

# Graficar el error de reconstrucción para cada imagen
plt.figure()
for i, errors in enumerate(errors_per_image):
    plt.plot(range(1, len(S) + 1), errors, label=f'Imagen {i+1}')
plt.xlabel('Número de dimensiones d')
plt.ylabel('Error de reconstrucción relativo')
plt.title('Error de reconstrucción por imagen en función del número de dimensiones')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

#Segunda parte del ejercicio
"""Utilizando esta ultima representación aprendida con el dataset 2 ¿Qué error de reconstrucción 
obtienen si utilizan la misma compresión (con la misma base de d dimensiones obtenida del dataset 2)
para las imagenes dataset_imagenes1.zip?"""


#funcion para obtener la matriz utilizada para reducir dimensión en PCA
#obtenemos el V_5 utilizado para la compresión con SVD del dataset_2
Vt_8 = fa.obtener_matriz_de_proyeccion(X_dataset2, d_min)

#Cargar las imágenes del dataset 1
from Ejercicio2_1 import X as X_dataset1

#Reducir la dimensionalidad de las imágenes del dataset 1 con V_5
X_dataset1_reducido = np.dot(np.dot(X_dataset1, Vt_8.T),Vt_8)

#Calcular el error de reconstrucción CON la norma de FROBENIUS
error_de_reconstruccion_dataset1 = fa.norma_de_Frobenius(X_dataset1 - np.dot(X_dataset1_reducido, Vt_8))/fa.norma_de_Frobenius(X_dataset1)

print(f"El error de reconstrucción para las imágenes del dataset 1 utilizando la compresión con d=5 del dataset 2 es: {error_de_reconstruccion_dataset1}")

#graficamos las imágenes con dimension reducida del dataset 1
plt.figure()
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(X_dataset1_reducido[i].reshape(p, p), cmap='gray')
    plt.axis('off')
plt.suptitle('Imágenes reconstruidas con d=5 del dataset 1')
plt.show()
