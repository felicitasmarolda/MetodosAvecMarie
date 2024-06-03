#Ejercicio 2.1
import numpy as np
import matplotlib.pyplot as plt

from Ejercicio2_2 import X
U, S, Vt = np.linalg.svd(X, full_matrices=False)

fig, axs = plt.subplots(1,3)
v1 = Vt[0]
v1 = v1.reshape(28,28)
axs[0].imshow(v1, cmap='gray')
axs[0].axis('off')
axs[0].set_title('Primer vector singular')
v2 = Vt[1]
v2 = v2.reshape(28,28)
axs[1].imshow(v2, cmap='gray')
axs[1].axis('off')
axs[1].set_title('Segundo vector singular')
v3 = Vt[2]
v3 = v3.reshape(28,28)
axs[2].imshow(v3, cmap='gray')
axs[2].axis('off')
axs[2].set_title('Tercer vector singular')
plt.show()

