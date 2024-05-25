#trabajo práctico 3 de MNyO

import numpy as np
import matplotlib.pyplot as plt
import funciones_auxiliares as fa

#Cargar el dataset X desde el archivo dataset.csv. en la carpeta dataset
X = np.loadtxt('dataset.csv', delimiter=',', skiprows=1)


# Descomponemos  X en sus valores singulares
U, S, Vt = np.linalg.svd(X)


#TODO descomentar
# Calculamos la matriz de similitud de X
W = fa.matriz_de_similitud(X, 10)

# Graficamos la matriz de similitud de X
# plt.figure()
# plt.imshow(W)
# plt.colorbar()
# plt.title('Matriz de similitud de X')
# plt.show()



#Queremos calcular X_k, con U_k, S_k y Vt_k y su matriz de similitud


#TODO descomentar
#calculamos X_k con k=2
U_2, S_2, Vt_2 = fa.calcular_X_k(U, S, Vt, 2)

# #buscamos X_2
# X_2 = np.dot(U_2, np.dot(np.diag(S_2), Vt_2))

# #calculamos la matriz de similitud de X_2
# W_2 = matriz_de_similitud(X_2, 10)

# #graficamos la matriz de similitud de X_2
# plt.figure()
# plt.imshow(W_2)
# plt.colorbar()
# plt.title('Matriz de similitud de X_2')
# plt.show()

#calculamos X_k con k=6
U_6, S_6, Vt_6 = fa.calcular_X_k(U, S, Vt, 6)

#TODO descomentar
# #calculamos X_6
# X_6 = np.dot(U_6, np.dot(np.diag(S_6), Vt_6))

# #calculamos la matriz de similitud de X_6
# W_6 = matriz_de_similitud(X_6, 100)

# #graficamos la matriz de similitud de X_6
# plt.figure()
# plt.imshow(W_6)
# plt.colorbar()
# plt.title('Matriz de similitud de X_6')
# plt.show()

#calculamos X_k con k=10
U_10, S_10, Vt_10 = fa.calcular_X_k(U, S, Vt, 10)

#TODO descomentar
# # #calculamos X_10
# X_10 = np.dot(U_10, np.dot(np.diag(S_10), Vt_10))


# #calculamos la matriz de similitud de X_10
# W_10 = matriz_de_similitud(X_10, 100)

# #graficamos la matriz de similitud de X_10
# plt.figure()
# plt.imshow(W_10)
# plt.colorbar()
# plt.title('Matriz de similitud de X_10')
# plt.show()

#HACEMOS PCA
#calculamos la matriz de covarianza de X
C = fa.matriz_de_covarianza(X)

#calculamos los componentes principales de X
Z = fa.PCA(X, 2)

#graficamos los componentes principales de X
plt.figure()
plt.scatter(Z[:, 0], Z[:, 1])
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.title('Componentes principales de X')
plt.show()


