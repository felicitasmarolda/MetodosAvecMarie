#Ejercicio 1.1


import numpy as np
import matplotlib.pyplot as plt
import funciones_auxiliares as fa

X = np.loadtxt('dataset.csv', delimiter=',', skiprows=1)
X = X[:, 1:]
Y = np.loadtxt('y.txt', delimiter=',')

def main():
    # Muestro la matriz X en un mapa de calor
    plt.figure()
    plt.imshow(X, cmap='magma')
    plt.title('Matriz X')
    plt.xlabel('Características')
    plt.ylabel('Muestras')
    plt.colorbar(label='Valor')
    plt.axis('tight')
    plt.show()

    # Muestro los valores singulares de X en un gráfico de barras
    plt.figure()
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    plt.bar(range(1, len(S) + 1), S, color='hotpink')
    plt.title('Valores singulares de X')
    plt.xlabel('Número de valor singular según su fila en S')
    plt.ylabel('Valores singulares de X')
    plt.bar(range(1, 3), S[:2], color='blue')
    plt.show()
    
    # Aplico PCA para d = 2 y muestro los resultados
    m, n = X.shape
    Z = fa.PCA(X, n)
    Z_2 = fa.PCA(X, 2)
    plt.scatter(Z[:, 0], Z[:, 1], c=Y)
    plt.xlabel('Componente principal 1')
    plt.ylabel('Componente principal 2')
    plt.title('Datos proyectados')
    plt.show()

    # Aplico PCA para d = 3 y muestro los resultados 
    X_3 = fa.PCA(X, 3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_3[:, 0], X_3[:, 1], X_3[:, 2], c=Y)
    ax.set_xlabel('Componente principal 1')
    ax.set_ylabel('Componente principal 2')
    ax.set_zlabel('Componente principal 3')
    plt.title('Datos proyectados en 3D')
    plt.show()

    # Proyección de los datos de X en el plano R2 utilizando el primer y tercer componente principal
    plt.figure()
    plt.scatter(X_3[:, 0], X_3[:, 2], c=Y)
    plt.xlabel('Componente principal 1')
    plt.ylabel('Componente principal 3')
    plt.title('Datos proyectados en el plano xz')
    plt.show()

    # Proyección de los datos de X en el plano R2 utilizando el segundo y tercer componente principal
    plt.figure()
    plt.scatter(X_3[:, 1], X_3[:, 2], c=Y)
    plt.xlabel('Componente principal 2')
    plt.ylabel('Componente principal 3')
    plt.title('Datos proyectados en el plano yz')
    plt.show()

    # Aplico PCA para 6 y 10 componentes
    Z_6 = fa.PCA(X, 6)
    Z_10 = fa.PCA(X, 10)

    # Calculo las matrices de similitud
    W_PCA = fa.matriz_de_similitud(Z, 10)
    W_PCA_2 = fa.matriz_de_similitud(Z_2, 10)
    W_PCA_6 = fa.matriz_de_similitud(Z_6, 10)
    W_PCA_10 = fa.matriz_de_similitud(Z_10, 10)

    # Muestro las matrices de similitud
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    im1 = axs[0, 0].imshow(W_PCA, cmap='viridis')
    axs[0, 0].set_title('Matriz de similitud de Z')
    im2 = axs[1, 1].imshow(W_PCA_2, cmap='viridis')
    axs[1, 1].set_title('Matriz de similitud de Z_2')
    im3 = axs[1, 0].imshow(W_PCA_6, cmap='viridis')
    axs[1, 0].set_title('Matriz de similitud de Z_6')
    im4 = axs[0, 1].imshow(W_PCA_10, cmap='viridis')
    axs[0, 1].set_title('Matriz de similitud de Z_10')
    plt.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im1, cax=cbar_ax)
    plt.show()


if __name__ == '__main__':
    main()