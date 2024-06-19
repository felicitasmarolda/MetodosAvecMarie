import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

n = 5
d = 100

# Definimos A y b aleatorios
A = np.random.randn(n,d)
b = np.random.randn(n,1)
# Sigma son los valores singulares de A
sigma = np.linalg.svd(A)[1] 
delta = (10**-2) * max(sigma)

# Calculamos el Hessiano de F1
H = 2 * A.T @ A
# Calculamos los autovalores de H
lamda_ = 2*np.max(np.linalg.svd(A)[1])**2
# Definimos s 
s = 1 / lamda_

# Definimos la función de costo F2 (se le agrega un término de regularización)
def F1(x, A, b):
    return (A @ x - b).T @ (A @ x - b)

def F2(x, A, b, delta):
    return (A @ x - b).T @ (A @ x - b) + delta * x.T @ x

def gradiente_F1(x, A, b):
    return 2 * A.T @ A @ x - 2 * A.T @ b

def gradiente_F2(x, A, b, delta):
    return 2 * A.T @ A @ x - 2 * A.T @ b + 2 * delta * x

# Definimos el metodo de descenso de gradiente iterativo para f1
def descenso_gradiente_F1(x1, s, cant_iteraciones, A, b):
    """
    Implementa el método de descenso de gradiente para la función F1.

    Parámetros:
    x1 : Vector inicial desde donde se comienza el descenso.
    s : Tasa de aprendizaje o tamaño de paso.
    cant_iteraciones : Cantidad de iteraciones a realizar.
    A : Matriz de coeficientes del sistema lineal.
    b : Vector de términos independientes del sistema lineal.

    Retorna:
    x : Vector resultante después de realizar el descenso de gradiente.
    costos : Lista con los costos (valores de F1) en cada iteración.
    x_values : Lista con los vectores x en cada iteración.
    """
    x = x1
    x_values = [x]
    costos = []
    for _ in range(cant_iteraciones):
        x = x - s * gradiente_F1(x, A, b)
        x_values.append(x)
        costos.append(F1(x, A, b))
    return x, costos, x_values

def descenso_gradiente_F2(x1, s, cant_iteraciones, A, b, delta):
    """
    Implementa el método de descenso de gradiente para la función F2.

    Parámetros:
    x1 : Vector inicial desde donde se comienza el descenso.
    s : Tasa de aprendizaje o tamaño de paso.
    cant_iteraciones : Cantidad de iteraciones a realizar.
    A : Matriz de coeficientes del sistema lineal.
    b : Vector de términos independientes del sistema lineal.
    delta : Parámetro de regularización.

    Retorna:
    x : Vector resultante después de realizar el descenso de gradiente.
    costos : Lista con los costos (valores de F1) en cada iteración.
    x_values : Lista con los vectores x en cada iteración.
    """
    x = x1
    x_values = [x]
    costos = []
    for _ in range(cant_iteraciones):
        x = x - s * gradiente_F2(x, A, b, delta)
        x_values.append(x)
        costos.append(F1(x, A, b))
    return x, costos, x_values

# Definimos la condición inicial aleatoriamente
x1 = np.random.rand(d,1)

# Definimos la cantidad de iteraciones
cant_iteraciones = 1000

# Calculamos el x que minimiza F1
x, costos, x_values1 = descenso_gradiente_F1(x1, s, cant_iteraciones, A, b)
print(f'X que minimiza F1: {x}')
costos = np.array(costos).flatten()

# Evaluamos F1 en x
print(F1(x, A, b))
# Evaluamos F1 en un x aleatorio
print(F1(np.random.rand(d,1), A, b))
#Calculamos el error de A*x-b
error = np.linalg.norm(A @ x - b)
print("Error de A*x-b: ", error)


# Hallamos el x que minimiza F2
x_2, costos_f2, x_values2 = descenso_gradiente_F2(x1, s, cant_iteraciones, A, b, delta)

costos_f2 = np.array(costos_f2).flatten()

# Calculamos el error de A*x-b
error_2 = np.linalg.norm(A @ x_2 - b)
print("Error de A*x-b con regularización: ", error_2)


# Resolvemos el problema mediante SVD -----------------------------------
U, S, Vt = np.linalg.svd(A)

# Calculamos el x que minimiza F1
#Creamos s inversa 100x5
S_inv = np.zeros((d,n))
#Llenamos la matriz con los valores singulares
S_inv[:n,:n] = np.diag(1/sigma)
x_svd = Vt.T @ S_inv @ U.T @ b
#Calculamos el error de A*x-b
error_svd = np.linalg.norm(A @ x_svd - b)
print("Error de A*x-b con SVD: ", error_svd)

# Graficamos los costo con los distintos métodos -------------------------
plt.plot(range(cant_iteraciones), costos, label='F1 evaluado en el X obtenido sin regularización', color='mediumorchid')
plt.plot(range(cant_iteraciones), costos_f2, label='F1 evaluado en el X obtenido con regularización', color='hotpink')
plt.axhline(y=F1(x_svd, A, b), color='green', label='F1 evaluado en el X obtenido con SVD')
plt.xlabel('Iteraciones')
plt.ylabel('Costo')
plt.yscale('log')
plt.legend()
plt.show()



# Vemos que ocurre con el costo cuando variamos delta -------------------
deltas = [10**-5,10**-4, 10**-3, 10**-2, 10**-1]
for delta in deltas:
    x1_copy = np.copy(x1)
    x_2, costos_f2, x = descenso_gradiente_F2(x1, s, cant_iteraciones, A, b, delta)
    costos_f2 = np.array(costos_f2).flatten()
    plt.plot(range(cant_iteraciones), costos_f2, label=f'delta = {delta}')

plt.xlabel('Iteraciones')
plt.ylabel('Valor de F1')
plt.legend()
plt.yscale('log')
plt.show()


#Calculamos el error de A*x-b en cada iteracion
error_x_values1 = [np.linalg.norm(A @ x - b) for x in x_values1]
error_x_values2 = [np.linalg.norm(A @ x - b) for x in x_values2]
# Graficamos el error de A*x-b en cada iteracion
plt.plot(range(cant_iteraciones+1), error_x_values1, label='Error de A*x-b con F1', color='mediumorchid')
plt.plot(range(cant_iteraciones+1), error_x_values2, label='Error de A*x-b con F2', color='hotpink')
plt.xlabel('Iteraciones')
plt.ylabel('Error de A*x-b')
plt.yscale('log')
plt.legend()
plt.show()


# Calculamos la norma de x vs iteraciones


#graficamos para x y x_2 en cada iteracion
normas_x = [np.linalg.norm(x) for x in x_values1]
normas_x_2 = [np.linalg.norm(x) for x in x_values2]

plt.plot(range(cant_iteraciones+1), normas_x, label='norma de x obtenidas con F1', color='mediumorchid')
plt.plot(range(cant_iteraciones+1), normas_x_2, label='norma de x obtenida con F2', color='hotpink')
plt.xlabel('Iteraciones')
plt.ylabel('Norma de x')
plt.legend()
plt.show()


# calculamos el error relativo de la solucion con SVD como la solucion de verdad y la solucion obtenida con el descenso de gradiente contra las iteraciones
# Calculamos el error relativo de la solucion con SVD
def error_relativo(x, x_svd):
    return np.linalg.norm(x - x_svd) / np.linalg.norm(x_svd)

#graficamos el error relativo para cada iteracion con x y x_2
error_relativo_x_values1 = [error_relativo(x, x_svd) for x in x_values1]
error_relativo_x_values2 = [error_relativo(x, x_svd) for x in x_values2]

plt.plot(range(cant_iteraciones+1), error_relativo_x_values1, label='Error relativo de x obtenida con F1', color='mediumorchid')
plt.plot(range(cant_iteraciones+1), error_relativo_x_values2, label='Error relativo de x obtenida con F2', color='hotpink')
plt.xlabel('Iteraciones')
plt.ylabel('Error relativo de x')
plt.legend()
plt.show()
