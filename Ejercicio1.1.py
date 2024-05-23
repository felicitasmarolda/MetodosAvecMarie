#trabajo práctico 3 de MNyO

import numpy as np
import matplotlib.pyplot as plt

#Cargar el dataset X desde el archivo dataset.csv. en la carpeta dataset
X = np.loadtxt('dataset.csv', delimiter=',', skiprows=1)


# Descomponemos  X en sus valores singulares
U, S, Vt = np.linalg.svd(X)

def distancia_euclideana(xi, xj, sigma):
    """"
    Calcula la distancia euclideana entre dos puntos xi y xj
    Recibe:
        xi: vector de dimensión n
        xj: vector de dimensión n
        sigma: ancho de banda del kernel
    Devuelve:
        distancia euclideana entre xi y xj
    """
    return np.exp(-((np.linalg.norm(xi-xj))**2)/(2*(sigma**2)))

def matriz_de_similitud(X, sigma):
    """
    Calcula la matriz de similitud de X
    Recibe:
        X: matriz de datos de dimensión m x n
        sigma: ancho de banda del kernel
    Devuelve:
        W: matriz de similitud de dimensión m x m
    """
    m, n = X.shape
    W = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            W[i, j] = distancia_euclideana(X[i], X[j], sigma)
    return W

#TODO descomentar
# Calculamos la matriz de similitud de X
W = matriz_de_similitud(X, 10)

# Graficamos la matriz de similitud de X
plt.figure()
plt.imshow(W)
plt.colorbar()
plt.title('Matriz de similitud de X')
plt.show()



#Queremos calcular X_k, con U_k, S_k y Vt_k y su matriz de similitud
#funcion que calcula los k valores singulares mas grandes
def calcular_valores_singulares(S, k):
    """
    Calcula los k valores singulares más grandes
    Recibe:
        S: vector de valores singulares de dimensión m
        k: número de valores singulares a conservar
    Devuelve:
        S_k: vector de valores singulares truncado de dimensión k
    """
    return S[:k]

#funcion que devuelve U_k, S_k y Vt_k
def calcular_X_k(U, S, Vt, k):
    """
    Calcula la descomposición en valores singulares truncada de X
    Recibe:
        U: matriz de vectores singulares izquierdos de dimensión m x m
        S: vector de valores singulares de dimensión m
        Vt: matriz de vectores singulares derechos de dimensión n x n
        k: número de valores singulares a conservar
    Devuelve:
        U_k: matriz de vectores singulares izquierdos truncada de dimensión m x k
        S_k: vector de valores singulares truncado de dimensión k
        Vt_k: matriz de vectores singulares derechos truncada de dimensión k x n
    """
    U_k = U[:, :k]      # Tomamos las primeras k columnas de U
    #tomamos los k valores singulares mas grandes
    S_k = calcular_valores_singulares(S, k)
    Vt_k = Vt[:k, :]    # Tomamos las primeras k filas de Vt
    return U_k, S_k, Vt_k

#calculamos X_k con k=2
U_2, S_2, Vt_2 = calcular_X_k(U, S, Vt, 2)

#TODO descomentar
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
U_6, S_6, Vt_6 = calcular_X_k(U, S, Vt, 6)

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
U_10, S_10, Vt_10 = calcular_X_k(U, S, Vt, 10)

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