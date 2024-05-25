
import numpy as np
import matplotlib.pyplot as plt

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

#Funcion que calcule la matriz de varianzas (hace el promedio de cada columna de X) --> devuelve una matriz del mismo tamaño que x pero todas las filas son iguales y tenes en cada lugar de la columna el promedio de esa columna
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

def matriz_de_promedio_columnas(X):
    """
    Calcula la matriz de varianzas de X
    Recibe:
        X: matriz de datos de dimensión m x n
    Devuelve:
        V: matriz de varianzas de dimensión m x n
    """
    m, n = X.shape
    V = np.zeros((m, n))
    for i in range(n):
        V[:, i] = np.mean(X[:, i])
    return V

#x - promedio de x
def normalizar(X):
    """
    Normaliza los datos de X
    Recibe:
        X: matriz de datos de dimensión m x n
    Devuelve:
        X_norm: matriz de datos normalizados de dimensión m x n
    """
    # Calculamos el promedio de cada columna de X
    promedio = matriz_de_promedio_columnas(X)
    # Restamos el promedio a cada columna de X
    X_norm = X - promedio
    return X_norm

# calculamos la matriz de covarianza
def matriz_de_covarianza(X):
    """
    Calcula la matriz de covarianza de X
    Recibe:
        X: matriz de datos de dimensión m x n
    Devuelve:
        C: matriz de covarianza de dimensión n x n
    """
    m, n = X.shape
    # Calculamos la matriz de covarianza
    C = np.dot(X.T, X) / m
    return C

# Hacer PCA
def PCA(X, k):
    """
    Realiza el análisis de componentes principales de X
    Recibe:
        X: matriz de datos de dimensión m x n
        k: número de componentes principales a conservar
    Devuelve:
        Z: matriz de datos proyectados de dimensión m x k
    """
    # Normalizamos los datos
    X_norm = normalizar(X)
    # Calculamos la matriz de covarianza
    C = matriz_de_covarianza(X_norm)
    # Calculamos la descomposición en valores singulares de la matriz de covarianza
    U, S, Vt = np.linalg.svd(C)
    # Calculamos la proyección de los datos
    U_k, S_k, Vt_k = calcular_X_k(U, S, Vt, k)
    # resultadi final deberia ser X_norm * V_k
    Z = np.dot(X_norm, Vt_k.T)

    return Z

