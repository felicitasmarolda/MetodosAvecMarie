#trabajo práctico 3 de MNyO

import numpy as np
import matplotlib.pyplot as plt
import funciones_auxiliares as fa

X = np.loadtxt('dataset.csv', delimiter=',', skiprows=1)
X = X[:, 1:]
Y = np.loadtxt('y.txt', delimiter=',')

def main():
    plt.figure()
    plt.imshow(X, cmap='magma')
    plt.title('Matriz X')
    plt.xlabel('Features')
    plt.ylabel('Muestras')
    plt.colorbar(label='Valor')
    plt.axis('tight')
    plt.show()
    
    m, n = X.shape
    Z = fa.PCA(X, n)

    Z_2 = fa.PCA(X, 2)

    plt.scatter(Z[:, 0], Z[:, 1], c=Y)
    plt.xlabel('Componente principal 1')
    plt.ylabel('Componente principal 2')
    plt.title('Datos proyectados')
    plt.show()

    X_3 = fa.PCA(X, 3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_3[:, 0], X_3[:, 1], X_3[:, 2], c=Y)
    ax.set_xlabel('Componente principal 1')
    ax.set_ylabel('Componente principal 2')
    ax.set_zlabel('Componente principal 3')
    plt.title('Datos proyectados en 3D')
    plt.show()

    plt.figure()
    plt.scatter(X_3[:, 0], X_3[:, 2], c=Y)
    plt.xlabel('Componente principal 1')
    plt.ylabel('Componente principal 3')
    plt.title('Datos proyectados en el plano xz')
    plt.show()

    plt.figure()
    plt.scatter(X_3[:, 1], X_3[:, 2], c=Y)
    plt.xlabel('Componente principal 2')
    plt.ylabel('Componente principal 3')
    plt.title('Datos proyectados en el plano yz')
    plt.show()


    # Z_6 = fa.PCA(X, 6)
    # Z_10 = fa.PCA(X, 10)

    # #hacer las matrices de similaridad del PCA
    # #TODO descomentar
    # W_PCA = fa.matriz_de_similitud(Z, 10)
    # W_PCA_2 = fa.matriz_de_similitud(Z_2, 10)
    # W_PCA_6 = fa.matriz_de_similitud(Z_6, 10)
    # W_PCA_10 = fa.matriz_de_similitud(Z_10, 10)


    # # Crear una figura y una cuadrícula de subplots 2x2
    # fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # # Graficar la matriz de similitud de Z en el primer subplot
    # im1 = axs[0, 0].imshow(W_PCA, cmap='viridis')
    # axs[0, 0].set_title('Matriz de similitud de Z')

    # # Graficar la matriz de similitud de Z_2 en el segundo subplot
    # im2 = axs[1, 1].imshow(W_PCA_2, cmap='viridis')
    # axs[1, 1].set_title('Matriz de similitud de Z_2')

    # # Graficar la matriz de similitud de Z_6 en el tercer subplot
    # im3 = axs[1, 0].imshow(W_PCA_6, cmap='viridis')
    # axs[1, 0].set_title('Matriz de similitud de Z_6')

    # # Graficar la matriz de similitud de Z_10 en el cuarto subplot
    # im4 = axs[0, 1].imshow(W_PCA_10, cmap='viridis')
    # axs[0, 1].set_title('Matriz de similitud de Z_10')

    # # Ajustar el espacio para la barra de colores
    # plt.subplots_adjust(right=0.8)

    # # Crear una barra de colores a la derecha de todos los subplots
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im1, cax=cbar_ax)

    # # Mostrar la figura
    # plt.show()


    #graficar los valores singulares de X en grafico de barras
    plt.figure()
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    plt.bar(range(1, len(S) + 1), S, color='hotpink')  # Modificado para graficar las raíces cuadradas de los elementos de la diagonal de S y con color hot pink
    plt.title('Valores singulares de X')  # Modificado para reflejar el cambio en el título
    plt.xlabel('Numero de valor singular según su fila en S')
    plt.ylabel('Valores singulares de X')  # Modificado para reflejar el cambio en la etiqueta del eje y
    
    # Pintar los dos primeros valores singulares de color celeste
    plt.bar(range(1, 3), S[:2], color='blue')
    
    plt.show()

    #FALTA
    # probar porque elegimos el valor de sigma

    # agarramos de las d dimanesiones mas importantes agarramos sus autovetores (lo que vienen de los valore ssingulares) y nos fijamos el numero mas grande de todos, nos fijamos en que posicion de su vector esta y esa posicion va a ser la dimension que mass dice


if __name__ == '__main__':
    main()