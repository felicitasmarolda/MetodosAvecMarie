#trabajo práctico 3 de MNyO

import numpy as np
import matplotlib.pyplot as plt

"""En el archivo dataset.csv se encuentra el dataset X. Este contiene un conjunto de n muestras que fueron
medidas a través de un sensor
{x1, x2, . . . , xi, . . . , xn}
con xi ∈ Rp (X es por lo tanto una matriz de n×p dimensiones). Si bien el conjunto tiene, a priori, dimensión
alta, es de interés entender visualmente como se distribuyen las muestras. Suponemos que las muestras no
se distribuyen uniformemente en el espacio Rp, por lo que podremos encontrar grupos de muestras (clusters)
con alta similaridad entre sí. La similaridad entre un par de muestras xi, xj se puede medir utilizando una
función no-lineal de su distancia euclidiana
K (xi, xj) = exp
 
−
∥xi − xj∥2
2
2σ2
!
,
para algún valor de σ.
Como la dimensionalidad inicial del dataset es muy alta y se supone que algunas dimensiones son mas
ruidosas que otras en las muestras, va a ser conveniente trabajar en un espacio de dimensión reducida d.
Para hacer esto hay que realizar una descomposición de X en sus valores singulares, reducir la dimensión de
esta representación, y luego trabajar con los vectores x proyectados al nuevo espacio reducido Z, es decir
z = V T
d x. Realizar los puntos anteriores para d = 2, 6, 10, y p. ¿Para qué elección de d resulta más
conveniente hacer el análisis? ¿Cómo se conecta esto con los valores singulares de X? ¿Qué conclusiones
puede sacar al respecto?
1. Determinar la similaridad par-a-par entre muestras en el espacio de dimension X y en el espacio
de dimensión reducida d para distintos valores de d utilizando PCA. Comparar estas medidas de
similaridad. Ayuda: ver de utilizar una matriz de similaridad para visualizar todas las similaridades
par-a-par juntas."""

#Cargar el dataset X desde el archivo dataset.csv. en la carpeta dataset
X = np.loadtxt('dataset.csv', delimiter=',', skiprows=1)


#realizar una descomposición de X en sus valores singulares
U, E, Vt = np.linalg.svd(X)

#obtener valores singulares de la diagonal de E en una lisra
valores_singulares = np.diag(E)

#hacer una funcion que obtenga los d elementos mayores de los valores singulares
def obtener_d_elementos_mayores(d, valores_singulares):
    valores_singulares_d = np.zeros_like(valores_singulares)
    for i in range(d):
        valores_singulares_d[i] = valores_singulares[i]
    return valores_singulares_d

#hacer una funcion que reescriba E como Ed que es E pero de tamaño dxd con los d valore singulares mas grandes como diagonal y 0 en el resto
def obtener_Ed(d, valores_singulares):
    Ed = np.zeros((d, d))
    for i in range(d):
        Ed[i, i] = valores_singulares[i]
    return Ed

#reescribimos Vt como Vtd que es Vt pero de tamaño dxp, mantiene sus primeras d filas
def obtener_Vtd(d, Vt):
    Vtd = np.zeros((d, Vt.shape[1]))
    for i in range(d):
        Vtd[i] = Vt[i]
    return Vtd

#reescribimos U como Ud que es U pero de tamaño nxp, mantiene sus primeras d columnas
def obtener_Ud(d, U):
    Ud = np.zeros((U.shape[0], d))
    for i in range(d):
        Ud[:, i] = U[:, i]
    return Ud

#hacemos una funcion que reciba xi y xj y sigma y devuelva la similaridad entre xi y xj (distancia euclideana)
def similaridad(xi, xj, sigma):
    return np.exp(-(np.linalg.norm(xi-xj))**2/(2*(sigma**2)))

#hacemos una funcion para calcular nuestra nueva matriz z=Vtd*x
def calcular_z(Vtd, X):
    return np.dot(Vtd, X.T)

#hacemos una funcion que reciba una matriz X y un sigma y devuelva la matriz de similaridades
def matriz_similaridades(X, sigma):
    n = X.shape[0]
    matriz_similaridades = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matriz_similaridades[i, j] = similaridad(X[i], X[j], sigma)
    return matriz_similaridades



