
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

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

def error_de_aproximacion(X_modificada,X_original):
    """
    Calcula el error de aproximación entre X_modificada y X_original
    Recibe:
        X_modificada: matriz de datos de dimensión m x n
        X_original: matriz de datos de dimensión m x n
    Devuelve:
        error: error de aproximación entre X_modificada y X_original
    """
    return np.linalg.norm(X_modificada - X_original)
#_____________________________________________________________________________________-
#Ejercicio 2
def cargar_y_transformar_imagen(ruta, p):
    if not os.path.isfile(ruta):
            print(f"El archivo {ruta} no se encuentra en el directorio especificado.")
            return None
    img = Image.open(ruta)
    img_array = np.array(img)
    img_vector = img_array.flatten()
    return img_vector

# def similaridad_entre_pares_de_imagenes(X):
#     """"""
#     #hacer svd
#     U, S, Vt = np.linalg.svd(X, full_matrices=False)
#     #valores de k
#     k_values = np.arange(1, 30)
#     #inicializamos la matriz de similaridad
#     matriz_similaridad = np.zeros((X.shape[0], X.shape[0], len(k_values)))
#     #inicializamos la matriz de similaridad
#     for i, k in enumerate(k_values):
#         U_k, S_k, Vt_k = calcular_X_k(U, S, Vt, k)
#         X_k = np.dot(U_k, np.dot(np.diag(S_k), Vt_k))
#         #calcular la matriz de similaridad
#         matriz_similaridad[:, :, i] = matriz_de_similitud(X_k, 10)
#     return matriz_similaridad

def similaridad_para_k_especifico(X, k):
    """"""
    # hacer svd
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    # calcular X_k
    U_k, S_k, Vt_k = calcular_X_k(U, S, Vt, k)
    X_k = np.dot(U_k, np.dot(np.diag(S_k), Vt_k))
    # calcular la matriz de similaridad
    matriz_similaridad = matriz_de_similitud(X_k, 10)
    return matriz_similaridad

def compresion(X, k):
    """
    Comprime una matriz X a k componentes utilizando SVD.

    Parámetros:
    X -- matriz de entrada
    k -- número de componentes a mantener

    Retorna:
    X_k -- matriz comprimida
    """
    
    # Calcular la descomposición de valores singulares de X
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Mantener solo las primeras k componentes
    U_k = U[:, :k]
    S_k = np.diag(S[:k])  # Convertir S_k en una matriz diagonal
    Vt_k = Vt[:k, :]

    # Calcular la matriz comprimida
    X_k = np.dot(U_k, np.dot(S_k, Vt_k))

    return X_k
