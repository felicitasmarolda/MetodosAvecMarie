#Ejercicio 2.3

import numpy as np
import matplotlib.pyplot as plt
import funciones_auxiliares as fa

from Ejercicio2_2 import X

X_15 = fa.compresion(X,15)
X_10 = fa.compresion(X,10)
X_5 = fa.compresion(X,5)
X_1 = fa.compresion(X,1)

sigma = 800
max_simil_X = fa.matriz_de_similitud(X,sigma)
mat_simil_15 = fa.matriz_de_similitud(X_15,sigma)
mat_simil_10 = fa.matriz_de_similitud(X_10,sigma) 
mat_simil_5 = fa.matriz_de_similitud(X_5,sigma) 
mat_simil_1 = fa.matriz_de_similitud(X_1,sigma) 

fig1, ax1 = plt.subplots(figsize=(6, 4))
im3 = ax1.imshow(mat_simil_1, cmap='viridis')
fig1.colorbar(im3, ax=ax1)
plt.show()

fig2, axes = plt.subplots(1, 4, figsize=(26, 6))

im2 = axes[0].imshow(mat_simil_5, cmap='viridis')
axes[0].set_title('Dimensión 5')
fig2.colorbar(im2, ax=axes[0])

im1 = axes[1].imshow(mat_simil_10, cmap='viridis')
axes[1].set_title('Dimensión 10')
fig2.colorbar(im1, ax=axes[1])

im5 = axes[2].imshow(mat_simil_15, cmap='viridis')
axes[2].set_title('Dimensión 15')
fig2.colorbar(im5, ax=axes[2])

im4 = axes[3].imshow(max_simil_X,cmap='viridis')
axes[3].set_title('X original')
fig2.colorbar(im4, ax=axes[3])


for ax in axes:
    ax.set_xticks(range(0, len(mat_simil_10), 5))
    ax.set_yticks(range(0, len(mat_simil_10), 5))

plt.tight_layout(pad=5.0)
plt.show()

plt.figure()

plt.subplot(2, 2, 1)
plt.imshow(X[3].reshape(28, 28), cmap='gray')
plt.title('Imagen 3 original')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(X_5[3].reshape(28, 28), cmap='gray')
plt.title('Imagen 3 con dimensión 5')
plt.axis('off')

plt.subplot(2, 2, 3) 
plt.imshow(X[15].reshape(28, 28), cmap='gray')
plt.title('Imagen 15 original')
plt.axis('off')

plt.subplot(2, 2, 4) 
plt.imshow(X_5[15].reshape(28, 28), cmap='gray')
plt.title('Imagen 15 con dimensión 5')
plt.axis('off')

plt.tight_layout()
plt.show()