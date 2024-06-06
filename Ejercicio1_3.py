#Ejercicio 1.3

import numpy as np
import matplotlib.pyplot as plt
import funciones_auxiliares as fa
from Ejercicio1_1 import X, Y

# Centro Y
Y = Y - np.mean(Y)

# Realizo SVD de X
U, S, Vt = np.linalg.svd(X, full_matrices=False)

# Realizo PCA para distintas dimensiones d
x_k_errores_beta = []
for i in range(1, min(X.shape)+1):
    # Calculo X_k
    X_k = fa.PCA(X, i)

    # Realizo SVD de X_k
    U_k, S_k, Vt_k = np.linalg.svd(X_k, full_matrices=False)
    
    # Calculo la inversa de S_k
    S_k_inv = np.diag(np.where(S_k != 0, 1/S_k, 0))

    # Calculo la pseudoinversa de X_k
    temp = np.dot(Vt_k.T, S_k_inv)
    x_k_pseudoinversa = np.dot(temp, U_k.T)
    
    # Calculo beta
    beta = np.dot(x_k_pseudoinversa, Y)
    
    # Calculo el error de predicción
    error = np.linalg.norm((np.dot(X_k, beta) - Y))
    x_k_errores_beta.append(error)

# Grafico los errores de predicción
plt.plot(range(1, min(X.shape)+1), x_k_errores_beta, color='hotpink')
plt.xlabel('Dimensión d')
plt.ylabel('Error de predicción')
plt.title('Error de predicción vs Dimensión d')
plt.grid(True)
plt.show()

# Calculo el hiperplano beta 
X_k = fa.PCA(X, 2)
U_k, S_k, Vt_k = np.linalg.svd(X_k, full_matrices=False)
S_k_inv = np.diag(np.where(S_k != 0, 1/S_k, 0))
temp = np.dot(Vt_k.T, S_k_inv)
x_k_pseudoinversa = np.dot(temp, U_k.T)
beta = np.dot(x_k_pseudoinversa, Y)

# Grafico los datos en 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_k[:, 0], X_k[:, 1], Y, c=Y)
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