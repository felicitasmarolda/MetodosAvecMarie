#Ejercicio 1.2

import numpy as np
import matplotlib.pyplot as plt
import funciones_auxiliares as fa
# Cargo los datos del ejercicio 1_1
from Ejercicio1_1 import X
from Ejercicio1_1 import Y


# Realizo la descomposición en valores singulares (SVD) de X
U, S, Vt = np.linalg.svd(X, full_matrices=False)

# Selecciono los dos autovectores más importantes
autovector = Vt[0]
autovector2 = Vt[1]

# Grafico los autovectores en una misma figura pero en diferentes subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].bar(range(1, len(autovector) + 1), autovector, color='blue')
axs[0].set_title('Autovector 1')
axs[0].set_xlabel('Número de feature')
axs[0].set_ylabel('Peso del feature en el autovector')
axs[1].bar(range(1, len(autovector2) + 1), autovector2, color='blue')
axs[1].set_title('Autovector 2')
axs[1].set_xlabel('Número de feature')
axs[1].set_ylabel('Peso del feature en el autovector')
plt.show()

# Busco los 5 valores absolutos más grandes de los autovectores
def find_max_AVE_position(autovector, autovector2):
    # Creo un diccionario con los valores absolutos de los autovectores ordenados y sus posiciones en el autovector original
    max_values = {i: abs(autovector[i]) for i in range(len(autovector))}
    max_values2 = {i: abs(autovector2[i]) for i in range(len(autovector2))}
    max_values = dict(sorted(max_values.items(), key=lambda item: item[1], reverse=True))
    max_values2 = dict(sorted(max_values2.items(), key=lambda item: item[1], reverse=True))
    return max_values, max_values2

max_values, max_values2 = find_max_AVE_position(autovector, autovector2)
max_values = list(max_values.items())[:6]
max_values2 = list(max_values2.items())[:6]

# Imprimo los 5 valores más grandes y sus posiciones para cada autovector
print(max_values)
print(max_values2)