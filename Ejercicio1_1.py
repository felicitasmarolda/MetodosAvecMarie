#trabajo práctico 3 de MNyO

import numpy as np
import matplotlib.pyplot as plt
import funciones_auxiliares as fa

#Cargar el dataset X desde el archivo dataset.csv. en la carpeta dataset
X = np.loadtxt('dataset.csv', delimiter=',', skiprows=1)
X = X[:, 1:]

# Descomponemos  X en sus valores singulares
U, S, Vt = np.linalg.svd(X)


#TODO descomentar
# Calculamos la matriz de similitud de X
# W = fa.matriz_de_similitud(X, 10)

# Graficamos la matriz de similitud de X
# plt.figure()
# plt.imshow(W)
# plt.colorbar()
# plt.title('Matriz de similitud de X')
# plt.show()

# Queremos calcular X_k, con U_k, S_k y Vt_k y su matriz de similitud


#TODO descomentar
#calculamos X_k con k=2
U_2, S_2, Vt_2 = fa.calcular_X_k(U, S, Vt, 2)

# #buscamos X_2
X_2 = np.dot(U_2, np.dot(np.diag(S_2), Vt_2))

# #calculamos la matriz de similitud de X_2
#W_2 = fa.matriz_de_similitud(X_2, 10)

# #graficamos la matriz de similitud de X_2
# plt.figure()
# plt.imshow(W_2)
# plt.colorbar()
# plt.title('Matriz de similitud de X_2')
# plt.show()

#calculamos X_k con k=6
U_6, S_6, Vt_6 = fa.calcular_X_k(U, S, Vt, 6)

#TODO descomentar
#calculamos X_6
X_6 = np.dot(U_6, np.dot(np.diag(S_6), Vt_6))

# #calculamos la matriz de similitud de X_6
#W_6 = fa.matriz_de_similitud(X_6, 100)

# #graficamos la matriz de similitud de X_6
# plt.figure()
# plt.imshow(W_6)
# plt.colorbar()
# plt.title('Matriz de similitud de X_6')
# plt.show()

#calculamos X_k con k=10
U_10, S_10, Vt_10 = fa.calcular_X_k(U, S, Vt, 10)

#TODO descomentar 
# #calculamos X_10
X_10 = np.dot(U_10, np.dot(np.diag(S_10), Vt_10))


# #calculamos la matriz de similitud de X_10
#W_10 = fa.matriz_de_similitud(X_10, 100)

# #graficamos la matriz de similitud de X_10
# plt.figure()
# plt.imshow(W_10)
# plt.colorbar()
# plt.title('Matriz de similitud de X_10')
# plt.show()

#HACEMOS PCA
#TODO descomentar
#calculamos los componentes principales de X
m, n = X.shape
Z = fa.PCA(X, n)

# calculamos los componentes principales de X_2
Z_2 = fa.PCA(X_2, 2)

#TODO descomentar
# #Graficamos los componentes principales de X_2
# plt.figure()
# plt.scatter(Z_2[:, 0], Z_2[:, 1])
# plt.title('Componentes principales de X_2')
# plt.show()

# Z_6 = fa.PCA(X_6, 6)
# Z_10 = fa.PCA(X, 10)

# #hacer las matrices de similaridad del PCA
# #TODO descomentar
# W_PCA = fa.matriz_de_similitud(Z, 10)
# W_PCA_2 = fa.matriz_de_similitud(Z_2, 10)
# W_PCA_6 = fa.matriz_de_similitud(Z_6, 10)
# W_PCA_10 = fa.matriz_de_similitud(Z_10, 10)


# # Crear una figura y una cuadrícula de subplots 2x2
# fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# # Graficar la matriz de similitud de Z en el primer subplot
# im1 = axs[0, 0].imshow(W_PCA, cmap='viridis')
# axs[0, 0].set_title('Matriz de similitud de Z')

# # Graficar la matriz de similitud de Z_2 en el segundo subplot
# im2 = axs[0, 1].imshow(W_PCA_2, cmap='viridis')
# axs[0, 1].set_title('Matriz de similitud de Z_2')

# # Graficar la matriz de similitud de Z_6 en el tercer subplot
# im3 = axs[1, 0].imshow(W_PCA_6, cmap='viridis')
# axs[1, 0].set_title('Matriz de similitud de Z_6')

# # Graficar la matriz de similitud de Z_10 en el cuarto subplot
# im4 = axs[1, 1].imshow(W_PCA_10, cmap='viridis')
# axs[1, 1].set_title('Matriz de similitud de Z_10')

# # Ajustar el espacio para la barra de colores
# plt.subplots_adjust(right=0.8)

# # Crear una barra de colores a la derecha de todos los subplots
# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
# fig.colorbar(im1, cax=cbar_ax)

# # Mostrar la figura
# plt.show()

#hacer una fórmula que calcule el error de aproximación de X_k
#TODO descomentar

#calcular y graficar este error para distintos valres de k (legando hasta k muy grande)
error = []
for k in range(1, 100):
    U_k, S_k, Vt_k = fa.calcular_X_k(U, S, Vt, k)
    X_k = np.dot(U_k, np.dot(np.diag(S_k), Vt_k))
    error.append(fa.error_de_aproximacion(X, X_k))

#graficar el error de aproximación de X_k
plt.figure()
plt.plot(range(1, 100), error)
plt.title('Error de aproximación de X_k')
plt.xlabel('k')
plt.ylabel('Error')
plt.show()

#FALTA
 # graficar los de los valores singulares y el porcentaje de noseque (creo que es parecido a lo que hicimos en el 2 con k)
 # probar porque elegimos el valor de sigma

