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

# Definimos la función de costo F2 (se le agrega un término de regularización)
def F(x, A, b, delta=0):
    return (A @ x - b).T @ (A @ x - b) + delta * np.linalg.norm(x, 2)**2

# Definimos el gradiente de F1
def gradiente_de_F(x, A, b, delta):
    return 2 * A.T @ A @ x - 2 * A.T @ b + 2 * delta * x

x_values = []

# Definimos el metodo de descenso de gradiente iterativo para f1
def descenso_gradiente(x1, s, cant_iteraciones, A, b, F, delta=0) -> tuple:
    costos = [] 
    x = x1
    x_values.append(x.copy()) 
    for i in range(cant_iteraciones):
        x.real = x - s * gradiente_de_F(x, A, b, delta)
        costo = F(x, A, b, delta)
        costos.append(float(costo))
        x_values.append(x.copy())
    return x, costos


# Definimos la condición inicial aleatoriamente
x1 = np.random.rand(d,1)

# Definimos la cantidad de iteraciones
cant_iteraciones = 1000

# Calculamos el x que minimiza F1
x, costos = descenso_gradiente(x1, s, cant_iteraciones, A, b, F)
# print(x)

# Evaluamos F1 en x
print(F(x, A, b))
# Evaluamos F1 en un x aleatorio
print(F(np.random.rand(d,1), A, b))
#Calculamos el error de A*x-b
error = np.linalg.norm(A @ x - b)
print("Error de A*x-b: ", error)


# Hallamos el x que minimiza F2
x_2, costos_f2 = descenso_gradiente(x1, s, cant_iteraciones, A, b, F, delta)
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

# Grafico de barras comparando los errores de los tres métodos en la iteración final
fig, axs = plt.subplots(2)  # Crear una figura con dos subgráficos

# Primer subgráfico para el descenso de gradiente y SVD
axs[0].bar(["Descenso de gradiente", "SVD"], [error, error_svd], color=['mediumorchid', 'royalblue'])
axs[0].set_ylabel('Error')

# Segundo subgráfico para el error de regularización
axs[1].bar(["Descenso de gradiente con regularización"], [error_2], color=['hotpink'])
axs[1].set_ylabel('Error')

plt.tight_layout()  # Ajustar el layout para que no se superpongan los gráficos
plt.show()


# Le agregamos ruido a la matriz A y probamos las soluciones obtenidas con ruido N=1000 veces y calculamos el error promedio de cada solución
N = 1000
error_promedio = 0
error_promedio_2 = 0
error_promedio_svd = 0

def add_noise(A, b, noise_std):
    A_noise = A + np.random.normal(0, noise_std, A.shape)
    b_noise = b + np.random.normal(0, noise_std, b.shape)
    return A_noise, b_noise

# for i in range(N):
#     # Generamos A con ruido
#     # A_ruido = A + np.random.normal(0, 0.1, (n,d))
#     A_ruido, b_ruido = add_noise(A, b, 0.1)
#     error_promedio += np.linalg.norm(A_ruido @ x - b_ruido)
#     error_promedio_2 += np.linalg.norm(A_ruido @ x_2 - b_ruido)
#     error_promedio_svd += np.linalg.norm(A_ruido @ x_svd - b_ruido)

error_avg_F1 = 0
error_avg_F2 = 0
error_avg_SVD = 0

for _ in range(N):
    A_noisy, b_noisy = add_noise(A, b, 0.5)

    error_avg_F1 += np.linalg.norm(A_noisy @ x - b_noisy)
    error_avg_F2 += np.linalg.norm(A_noisy @ x_2 - b_noisy)
    error_avg_SVD += np.linalg.norm(A_noisy @ x_svd - b_noisy)



error_avg_F1 /= N
error_avg_F2 /= N
error_avg_SVD /= N

print("Error promedio de A*x-b con ruido en F1: ", error_avg_F1)
print("Error promedio de A*x-b con ruido en F2: ", error_avg_F2)
print("Error promedio de A*x-b con ruido en SVD: ", error_avg_SVD)
