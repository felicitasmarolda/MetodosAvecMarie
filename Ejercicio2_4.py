#Ejercicio 2.4

import numpy as np
import matplotlib.pyplot as plt
import os
import funciones_auxiliares as fa

# Defino la carpeta donde se encuentran las imágenes y las cargo
img_folder = 'datasets_imgs_02'
img_files = os.listdir(img_folder)
img_files = [f for f in img_files if f.endswith('.jpeg')]
p = 28
X_dataset2 = np.zeros((len(img_files), p**2))
for i, f in enumerate(img_files):
    img_vector = fa.cargar_y_transformar_imagen(os.path.join(img_folder, f), p)
    if img_vector is not None:
        X_dataset2[i] = img_vector

# Muestro algunas de las imágenes cargadas
plt.figure()
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(X_dataset2[i].reshape(p, p), cmap='gray')
    plt.axis('off')
plt.show()

# Realizo la descomposición en valores singulares de las imágenes
U, S, Vt = np.linalg.svd(X_dataset2, full_matrices=False)
plt.figure()
plt.bar(range(1, 9), S[:8])
plt.title('Valores singulares de X')
plt.show()

# Defino una función para calcular el error de reconstrucción por imagen
def calcular_error_por_imagen():
    error_por_imagen = []
    for i in range(1, 9):
        imagen = []
        for j in range(1, 9):
            S_d = np.zeros((U.shape[0], Vt.shape[0]))
            np.fill_diagonal(S_d[:i, :i], S[:i])
            X_d = np.dot(U, np.dot(S_d, Vt))

            errores = np.linalg.norm(X_dataset2 - X_d, axis=1) / np.linalg.norm(X_dataset2, axis=1)
            imagen.append(errores[j-1])
        error_por_imagen.append(imagen)
    return error_por_imagen

# Muestro el error de reconstrucción por imagen
plt.figure()

error_por_imagen = calcular_error_por_imagen()

for i in range(8):
    errores_imagen_i = [error_por_imagen[j][i] for j in range(8)]
    plt.plot(range(1, 9), errores_imagen_i, label=f'Imagen {i+1}')

plt.axhline(y=0.1, color='r', linestyle='--')

plt.legend()

plt.xlabel('Dimensiones')
plt.ylabel('Error de reconstrucción')

plt.show()

# Calculo las dimensiones mínimas por imagen
dimensiones_minimas_por_imagen = []
for i in range(len(error_por_imagen)):
    for j in range(len(error_por_imagen[i])):
        if error_por_imagen[j][i] <= 0.1:
            dimensiones_minimas_por_imagen.append(j+1)
            break

print(dimensiones_minimas_por_imagen)

dimension_minima = max(dimensiones_minimas_por_imagen)   

dimension_maxima = max(dimensiones_minimas_por_imagen)

# Importo el dataset 1 y calculo el error de reconstrucción utilizando la compresión del dataset 2
from Ejercicio2_2 import X as X_dataset1
U,S,Vt = np.linalg.svd(X_dataset2)
Vt_8 = Vt[:8,:]
X_dataset1_reducido = X_dataset1 @ (Vt_8.T @ Vt_8)

error_de_reconstruccion_dataset1 = np.linalg.norm(X_dataset1 - X_dataset1_reducido) / np.linalg.norm(X_dataset1)

print(f"El error de reconstrucción para las imágenes del dataset 1 utilizando la compresión con d=8 del dataset 2 es: {error_de_reconstruccion_dataset1}")

# Muestro algunas de las imágenes del dataset 1 reducidas
fig, axs = plt.subplots(1, 19, figsize = (19,2))
for i in range(19):
    vector = X_dataset1_reducido[i].reshape(28,28)
    axs[i].imshow(vector, cmap='gray')
    axs[i].axis('off')
plt.show()

# Realizo la descomposición en valores singulares de las imágenes del dataset 2
U, S, Vt = np.linalg.svd(X_dataset2, full_matrices=False)
plt.figure()
plt.bar(range(1, 9), S[:8])
plt.title('Valores singulares de X')
plt.show()

# Calculo el error de reconstrucción en función del número de dimensiones
fro_norm_original = np.linalg.norm(X_dataset2, 'fro')

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

# Muestro el error de reconstrucción en función del número de dimensiones
plt.figure()
plt.plot(range(1, len(S) + 1), [fn / fro_norm_original for fn in fro_norms], marker='o')
plt.axhline(y=0.1, color='r', linestyle='--')
plt.xlabel('Número de dimensiones d')
plt.ylabel('Error de reconstrucción relativo')
plt.title('Error de reconstrucción en función del número de dimensiones')
plt.show()