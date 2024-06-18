import numpy as np
import matplotlib.pyplot as plt

n = 5
d = 100
# Definimos A y b aleatorios
# Generamos A de n filas y d columnas 
A = np.random.randn(n,d)
# Generamos b de n filas y 1 columna
b = np.random.randn(n,1)
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
    """CALCULA EL DESCENSO GRADIENTE DE UNA FUNCION F1"""
    x = x1
    x_values = [x]
    costos = []
    for _ in range(cant_iteraciones):
        x = x - s * gradiente_F1(x, A, b)
        x_values.append(x)
        costos.append(F1(x, A, b))
    return x, costos, x_values

def descenso_gradiente_F2(x1, s, cant_iteraciones, A, b, delta):
    """CALCULA EL DESCENSO GRADIENTE DE UNA FUNCION F2"""
    x = x1
    x_values = [x]
    costos = []
    for _ in range(cant_iteraciones):
        x = x - s * gradiente_F2(x, A, b, delta)
        x_values.append(x)
        costos.append(F2(x, A, b, delta))
    return x, costos, x_values

# Definimos la condición inicial aleatoriamente
x1 = np.random.rand(d,1)

# Definimos la cantidad de iteraciones
cant_iteraciones = 1000

# Calculamos el x que minimiza F1
x, costos, x_values1 = descenso_gradiente_F1(x1, s, cant_iteraciones, A, b)
# print(x)

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
deltas = [10**-5,10**-4, 10**-3, 10**-2, 10**-1]

for delta in deltas:
    x1_copy = np.copy(x1)
    x_2, costos_f2, x = descenso_gradiente_F2(x1, s, cant_iteraciones, A, b, delta)
    costos_f2 = np.array(costos_f2).flatten()
    plt.plot(range(cant_iteraciones), costos_f2, label=f'delta = {delta}')

plt.xlabel('Iteraciones')
plt.ylabel('Costo')
plt.legend()
plt.yscale('log')
plt.show()

# Graficamos las curvas de isocosto -------------------------------------------------
# Generar una malla para las curvas de iso-costos

# Hcaemos PCA para reducir la dimensión a 2
U, S, Vt = np.linalg.svd(A)
A_2 = A @ Vt.T[:,:2]
