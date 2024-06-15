import numpy as np
import matplotlib.pyplot as plt
import numpy as np

n = 5
d = 100
# Definimos A y b aleatorios
# Generamos A de n filas y d columnas 
A = np.random.rand(n,d)
# Generamos b de n filas y 1 columna
b = np.random.rand(n,1)
# Sigma son los valores singulares de A
sigma = np.linalg.svd(A)[1] #OJO
# definimos delta
delta = (10**-2) * max(sigma)

# Calculamos el Hessiano de F1, no de F2
H = 2 * A.T @ A + 2 * delta * np.eye(d)
# Calculamos los autovalores de H
lamda_ = np.linalg.eig(H)[0]
# Definimos s 
s = 1 / max(lamda_) if max(lamda_) > 0 else 1

# Definimos la función de costo F1
def F1(x, A, b, delta=0) -> float:
    return (A @ x - b).T @ (A @ x - b)

# Definimos la función de costo F2 (se le agrega un término de regularización)
def F2(x, A, b, delta):
    return F1(x, A, b) + delta * np.linalg.norm(x, 2)**2

# Definimos el gradiente de F1
def gradiente_de_F(x, A, b, delta):
    return 2 * A.T @ A @ x - 2 * A.T @ b + 2 * delta * x

# Definimos el metodo de descenso de gradiente iterativo para f1
def descenso_gradiente(x1, s, cant_iteraciones, A, b, F, delta=0) -> tuple:
    costos = [] 
    x = x1
    for i in range(cant_iteraciones):
        x.real = x - s * gradiente_de_F(x, A, b, delta)
        costo = F(x, A, b, delta).real
        costos.append(float(costo))
    return x, costos


# Definimos la condición inicial aleatoriamente
x1 = np.random.rand(d,1)

# Definimos la cantidad de iteraciones
cant_iteraciones = 1000

# Calculamos el x que minimiza F1
x, costos = descenso_gradiente(x1, s, cant_iteraciones, A, b, F1)
# print(x)

# Evaluamos F1 en x
print(F1(x, A, b))
# Evaluamos F1 en un x aleatorio
print(F1(np.random.rand(d,1), A, b))
#Calculamos el error de A*x-b
error = np.linalg.norm(A @ x - b)
print("Error de A*x-b: ", error)


# Hallamos el x que minimiza F2
x_2, costos_f2 = descenso_gradiente(x1, s, cant_iteraciones, A, b, F2, delta)
# Calculamos el error de A*x-b
error_2 = np.linalg.norm(A @ x_2 - b)
print("Error de A*x-b con regularización: ", error_2)

# print(x)
# Graficamos la función de costo F2 vs la cantidad de iteraciones, en el mismo gráfico que F1
plt.plot(range(cant_iteraciones), costos, label='F1', color='mediumorchid')
plt.plot(range(cant_iteraciones), costos_f2, label='F2', color='hotpink')
plt.xlabel('Iteraciones')
plt.ylabel('Costo')
plt.legend()
plt.show()

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

