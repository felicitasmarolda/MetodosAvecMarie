#Ejercicio 1.3

import numpy as np
import matplotlib.pyplot as plt
import os
import funciones_auxiliares as fa

"""Los datosX vienen acompañados de una variable dependiente respuesta o etiquetas llamada Y (archivo
y.txt) estructurada como un vector nx1 para cada muestra. Queremos encontrar el vector β y modelar
linealmente el problema que minimice la norma
∥X * β - y∥2
de manera tal de poder predecir con X*β = y lo mejor posible a las etiquetas y, es decir, minimizar el
error de predicción. Usando PCA, que dimensión d mejora la predicción? Cuales muestras son las de
mejor predicción con el mejor modelo? Resolviendo el problema de cuadrados mínimos en el espacio
original X, que peso se le asigna a cada dimensión original si observamos el vector β?"""

import numpy as np
import matplotlib.pyplot as plt
import os
import funciones_auxiliares as fa

# Cargar los datos
from Ejercicio1_1 import X
from Ejercicio1_1 import Y

# Centrar Y
Y = Y - np.mean(Y)


# Hacer SVD de X
U, S, Vt = np.linalg.svd(X, full_matrices=False)

# Hacemos PCA para distintas dimensiones d
# Hacemos PCA para distintas dimensiones d
x_k_errores_beta = []
for i in range(1, min(X.shape)+1):
    # Calcular X_k
    X_k = fa.PCA(X, i)
    

    # Hacer SVD de X_k
    U_k, S_k, Vt_k = np.linalg.svd(X_k, full_matrices=False)
    
    # Multiplicar V por S(inversa) por U.T

    #Calcular la inversa de S_k, si el autovalor es 0, la diagonal en esa posición es 0
    S_k_inv = np.diag(np.where(S_k != 0, 1/S_k, 0))

    temp = np.dot(Vt_k.T, S_k_inv)
    x_k_pseudoinversa = np.dot(temp, U_k.T)
    
    # Multiplicar la pseudoinversa por Y para calcular beta
    beta = np.dot(x_k_pseudoinversa, Y)
    
    # Calcular el error de predicción usando la norma 2
    error = np.linalg.norm((np.dot(X_k, beta) - Y))
    x_k_errores_beta.append(error)

# Graficar los errores de predicción para diferentes dimensiones
plt.plot(range(1, min(X.shape)+1), x_k_errores_beta, color='hotpink')
plt.xlabel('Dimensión d')
plt.ylabel('Error de predicción')
plt.title('Error de predicción vs Dimensión d')
plt.grid(True)
plt.show()

# Graficamos el hiperplano
# Calcular el hiperplano
X_k = fa.PCA(X, 2)
U_k, S_k, Vt_k = np.linalg.svd(X_k, full_matrices=False)
S_k_inv = np.diag(np.where(S_k != 0, 1/S_k, 0))
temp = np.dot(Vt_k.T, S_k_inv)
x_k_pseudoinversa = np.dot(temp, U_k.T)
beta = np.dot(x_k_pseudoinversa, Y)

# Graficar en la coordenada x el pirmer componente principal y en la coordenada y el segundo componente principal y el la z el valor de Y
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_k[:, 0], X_k[:, 1], Y, c=Y)
# grafique el hiperplano beta
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)
z = beta[0]*x + beta[1]*y
ax.plot_surface(x, y, z, alpha=0.3)
ax.set_xlabel('Componente principal 1')
ax.set_ylabel('Componente principal 2')
ax.set_zlabel('etiqueta Y')
plt.title('Datos proyectados en 3D')
plt.show()



# #multiplicamos V por S(inveras) por UT por Y
# beta = np.dot(np.dot(np.dot(Vt.T, np.diag(1/S)), U.T), Y)

# #Calculamos el error de predicción
# error = fa.error_de_prediccion(X, Y, beta)
# print(f"El error de predicción es: {error}")

# # #calcular X_k para calcular un nuevo beta pero con menor dimensionalidad
# # d = 10
# # U_k, S_k, Vt_k = fa.calcular_X_k(U, S, Vt, d)
# # beta_k = np.dot(np.dot(np.dot(Vt_k.T, np.diag(1/S_k)), U_k.T), Y)
# # error_k = fa.error_de_prediccion(X, Y, beta_k)
# # print(f"El error de predicción con d=1 es: {error_k}")

# errores = []
# dimensiones = range(1, min(X.shape)+1)
# for d in dimensiones:
#     U_k, S_k, Vt_k = fa.calcular_X_k(U, S, Vt, d)
#     beta_k = np.dot(np.dot(np.dot(Vt_k.T, np.diag(1/S_k)), U_k.T), Y)
#     error_k = fa.error_de_prediccion(X, Y, beta_k)
#     errores.append(error_k)

# # Encontrar la mejor dimensión d que minimiza el error de predicción
# mejor_d = dimensiones[np.argmin(errores)]
# print(f"La mejor dimensión d es: {mejor_d}")

# # Calcular beta y el error de predicción con la mejor dimensión d
# U_k, S_k, Vt_k = fa.calcular_X_k(U, S, Vt, mejor_d)
# beta_k = np.dot(np.dot(np.dot(Vt_k.T, np.diag(1/S_k)), U_k.T), Y)
# error_k = fa.error_de_prediccion(X, Y, beta_k)

# # Graficar los errores de predicción para diferentes dimensiones
# plt.plot(dimensiones, errores)
# plt.xlabel('Dimensión d')
# plt.ylabel('Error de predicción')
# plt.title('Error de predicción vs Dimensión d')
# plt.grid(True)
# plt.show()

# #Cuales muestras son las de mejor predicción con el mejor modelo?
# #comparemos los errores de cda coordeanda de Y con la predicción correspondiente, es deccir yi con X[i]*beta_k

# #mostra estos errores en un gráfico donde el eje x sea el sub indice de la muestra y el eje y sea el error de predicción

# errores_muestras = []
# for i in range(len(Y)):
#     error_muestra = fa.error_de_prediccion(X[i], Y[i], beta_k)
#     errores_muestras.append(error_muestra)

# plt.plot(range(len(Y)), errores_muestras)
# plt.xlabel('Muestra')
# plt.ylabel('Error de predicción')
# plt.title('Error de predicción por muestra')
# plt.grid(True)
# plt.show()

# #devolveme el menor valor de error de predicción con su indice, pone tambien una tupla con el valor de Y orginal y el valor de la predicción
# min_error = min(errores_muestras)
# indice_min_error = errores_muestras.index(min_error)
# print(f"El menor error de predicción es: {min_error}")
# print(f"El índice de la muestra con menor error de predicción es: {indice_min_error+1}")
# print(f"El valor de Y original es: {Y[indice_min_error]}")
# print(f"El valor de la predicción es: {np.dot(X[indice_min_error], beta_k)}")


#FALTA
#grafico del error con dimension despues de hcaer cuadrados mínimos --> decir lo que que el error baja con ma dimensionpero que no así le da "mucha libertad" a los valorees decir, al hacer que parezcan a los originales, no se adapta bien a otras situaciones ==> este grafico es muy limitante pero hay que hacerlo igual
