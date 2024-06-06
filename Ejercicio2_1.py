#Ejercicio 2.1

import numpy as np
import matplotlib.pyplot as plt
from Ejercicio2_2 import X

# Realizo la descomposición en valores singulares (SVD) de X
U, S, Vt = np.linalg.svd(X, full_matrices=False)

fig, axs = plt.subplots(1,3)

# Itero sobre las tres primeras componentes principales
for i in range(3):
    # Redimensiono la componente principal a una matriz de 28x28
    v = Vt[i].reshape(28,28)
    # Muestro la componente principal como una imagen en escala de grises
    axs[i].imshow(v, cmap='gray')
    axs[i].axis('off')

plt.show()