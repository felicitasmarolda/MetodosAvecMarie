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
# W = fa.matriz_de_similitud(X, 10)

# Graficamos la matriz de similitud de X
# plt.figure()
# plt.imshow(W)
# plt.colorbar()
# plt.title('Matriz de similitud de X')
# plt.show()



#Queremos calcular X_k, con U_k, S_k y Vt_k y su matriz de similitud


#TODO descomentar
#calculamos X_k con k=2
#U_2, S_2, Vt_2 = fa.calcular_X_k(U, S, Vt, 2)

# #buscamos X_2
#X_2 = np.dot(U_2, np.dot(np.diag(S_2), Vt_2))

# #calculamos la matriz de similitud de X_2
# W_2 = matriz_de_similitud(X_2, 10)

# #graficamos la matriz de similitud de X_2
# plt.figure()
# plt.imshow(W_2)
# plt.colorbar()
# plt.title('Matriz de similitud de X_2')
# plt.show()

#calculamos X_k con k=6
# U_6, S_6, Vt_6 = fa.calcular_X_k(U, S, Vt, 6)

#TODO descomentar
#calculamos X_6
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
#calculamos X_10
X_10 = np.dot(U_10, np.dot(np.diag(S_10), Vt_10))


# #calculamos la matriz de similitud de X_10
# W_10 = matriz_de_similitud(X_10, 100)

# #graficamos la matriz de similitud de X_10
# plt.figure()
# plt.imshow(W_10)
# plt.colorbar()
# plt.title('Matriz de similitud de X_10')
# plt.show()

#HACEMOS PCA
#TODO descomentar
# #calculamos los componentes principales de X
# m, n = X.shape
# Z = fa.PCA(X, n)

# # calculamos los componentes principales de X_2
# Z_2 = fa.PCA(X_2, 2)

#Graficamos los componentes principales de X_2
# plt.figure()
# plt.scatter(Z_2[:, 0], Z_2[:, 1])
# plt.title('Componentes principales de X_2')
# plt.show()

# #graficar la matriz Z
# plt.figure()
# plt.imshow(Z_2, aspect='auto')
# plt.colorbar()
# plt.title('Matriz Z_2')
# plt.show()


# #Calculo Z_6 los componentes principales de X_6
# Z_6 = fa.PCA(X_6, 6)
# # Graficamos la matriz Z_6
# plt.figure()
# plt.imshow(Z_6, aspect='auto')
# plt.colorbar()
# plt.title('Matriz Z_6')
# plt.show()

#Calculo Z_10 los componentes principales de X_10
Z_10 = fa.PCA(X, 10)
# Graficamos la matriz Z_10
plt.figure()
plt.imshow(Z_10, aspect='auto')
plt.colorbar()  
plt.title('Matriz Z_10')
plt.show()