import numpy as np
import matplotlib.pyplot as plt

# Función para agregar ruido a A y b
def add_noise(A, b, noise_std):
    A_noise = A + np.random.normal(0, noise_std, A.shape)
    b_noise = b + np.random.normal(0, noise_std, b.shape)
    return A_noise, b_noise

# Definimos A y b aleatorios
n = 5
d = 100
A = np.random.rand(n, d)
b = np.random.rand(n, 1)

# Sigma son los valores singulares de A
sigma = np.linalg.svd(A, compute_uv=False)
delta = (10**-2) * max(sigma)**2

# Calculamos el Hessiano de F1, no de F2
H = 2 * A.T @ A + 2 * delta * np.eye(d)
lamda_ = np.linalg.eigvalsh(H)
s = 1 / max(lamda_) if max(lamda_) > 0 else 1

# Definimos la función de costo F1
def F1(x, A, b):
    return np.linalg.norm(A @ x - b)**2

# Definimos la función de costo F2
def F2(x, A, b, delta):
    return F1(x, A, b) + delta * np.linalg.norm(x)**2

# Definimos el gradiente de F1
def gradiente_de_F(x, A, b):
    return 2 * A.T @ (A @ x - b)

# Definimos el método de descenso de gradiente
def descenso_gradiente(x1, s, cant_iteraciones, A, b, F, delta=0):
    costos = []
    trayectorias = [x1]  # Almacena la trayectoria de x
    x = x1
    for i in range(cant_iteraciones):
        x = x - s * gradiente_de_F(x, A, b)
        costo = F(x, A, b, delta)
        costos.append(costo)
        trayectorias.append(x.copy())
    return x, costos, np.array(trayectorias)

# Condición inicial aleatoria
x1 = np.random.rand(d, 1)

# Cantidad de iteraciones
cant_iteraciones = 1000

# Resolución de F1 y F2 sin ruido
x_F1, costos_F1, tray_F1 = descenso_gradiente(x1, s, cant_iteraciones, A, b, F1)
x_F2, costos_F2, tray_F2 = descenso_gradiente(x1, s, cant_iteraciones, A, b, F2, delta)

# Error de A*x-b sin regularización y con regularización
error_F1 = np.linalg.norm(A @ x_F1 - b)
error_F2 = np.linalg.norm(A @ x_F2 - b)

print(f"Error de A*x-b (F1): {error_F1}")
print(f"Error de A*x-b (F2): {error_F2}")

# Añadir ruido a A y b y calcular errores promedio
N = 1000
noise_std = 0.1

error_avg_F1 = 0
error_avg_F2 = 0

for _ in range(N):
    A_noisy, b_noisy = add_noise(A, b, noise_std)
    
    x_F1_noisy, _, tray_F1_noisy = descenso_gradiente(x1, s, cant_iteraciones, A_noisy, b_noisy, F1)
    x_F2_noisy, _, tray_F2_noisy = descenso_gradiente(x1, s, cant_iteraciones, A_noisy, b_noisy, F2, delta)
    
    error_avg_F1 += np.linalg.norm(A_noisy @ x_F1_noisy - b_noisy)
    error_avg_F2 += np.linalg.norm(A_noisy @ x_F2_noisy - b_noisy)

error_avg_F1 /= N
error_avg_F2 /= N

print(f"Error promedio de A*x-b con ruido en F1: {error_avg_F1}")
print(f"Error promedio de A*x-b con ruido en F2: {error_avg_F2}")

# Visualización de Trayectorias
plt.figure(figsize=(14, 6))

# Trayectorias de F1 sin ruido
plt.subplot(121)
plt.plot(tray_F1[:, 0, 0], tray_F1[:, 1, 0], label='Trayectoria F1 sin ruido', color='blue')
plt.xlabel('x[0]')
plt.ylabel('x[1]')
plt.title('Trayectoria de F1 sin ruido')
plt.legend()
plt.grid(True)

# Trayectorias de F2 con ruido
plt.subplot(122)
plt.plot(tray_F2[:, 0, 0], tray_F2[:, 1, 0], label='Trayectoria F2 con ruido', color='red')
plt.xlabel('x[0]')
plt.ylabel('x[1]')
plt.title('Trayectoria de F2 con ruido')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Graficar la función de costo
plt.figure(figsize=(10, 5))
plt.plot(range(cant_iteraciones), costos_F1, label='F1', color='mediumorchid')
plt.plot(range(cant_iteraciones), costos_F2, label='F2', color='hotpink')
plt.xlabel('Iteraciones')
plt.ylabel('Costo')
plt.legend()
plt.title('Función de Costo')
plt.grid(True)
plt.show()

# Comparar errores finales con SVD
U, S, Vt = np.linalg.svd(A)
S_inv = np.zeros((d, n))
S_inv[:n, :n] = np.diag(1 / sigma)
x_svd = Vt.T @ S_inv @ U.T @ b
error_svd = np.linalg.norm(A @ x_svd - b)
print(f"Error de A*x-b con SVD: {error_svd}")

# Graficar barras comparando errores
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

axs[0].bar(["Descenso de gradiente", "SVD"], [error_F1, error_svd], color=['mediumorchid', 'royalblue'])
axs[0].set_ylabel('Error')
axs[0].set_title('Error de A*x-b sin ruido')

axs[1].bar(["Descenso de gradiente con regularización"], [error_avg_F2], color=['hotpink'])
axs[1].set_ylabel('Error')
axs[1].set_title('Error de A*x-b con ruido')

plt.tight_layout()
plt.show()
