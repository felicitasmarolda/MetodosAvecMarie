#Ejercicio 1.2
 # agarramos de las d dimanesiones mas importantes agarramos sus autovetores (lo que vienen de los valore ssingulares) y nos fijamos el numero mas grande de todos, nos fijamos en que posicion de su vector esta y esa posicion va a ser la dimension que mass dice

import numpy as np
import matplotlib.pyplot as plt
import os
import funciones_auxiliares as fa

#Cargar los datos del ejercicio 1_1
from Ejercicio1_1 import X
from Ejercicio1_1 import Y



#las columnas del dataset representan las features 
# 1 te fijas los valores singulares mas imposrtantes porque son los qu emarcan el gran porcentaje de varianza de la matriz
# tenemos un autovector asociado que rerpresenta las features mas importantes y el peso que tiene cada feature sobre el valor singular
# graficas los features  (6 van a ser los q pesan mas q el resto) hay 3 y 3 que tienen el mismo valor abs y apuntan para la misma direccion


#Hacer SVD de X
U, S, Vt = np.linalg.svd(X, full_matrices=False)

#buscamos los 2 autovectores más importantes
autovector = Vt[0]
autovector2 = Vt[1]

# Graficar los autovectores en una misma figura pero distinos subplots
ig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].bar(range(1, len(autovector) + 1), autovector, color='blue')
axs[0].set_title('Autovector 1')
axs[0].set_xlabel('Número de feature')
axs[0].set_ylabel('Peso del feature en el autovector')
axs[1].bar(range(1, len(autovector2) + 1), autovector2, color='blue')
axs[1].set_title('Autovector 2')
axs[1].set_xlabel('Número de feature')
axs[1].set_ylabel('Peso del feature en el autovector')
plt.show()

                    

# buscamos  los 5 valores absolutos más grandes de los autovectores
def find_max_AVE_position(autovector, autovector2):
    #creo un diccionario con los valores absolutos de los autovectores ordenados y sus posiciones en el autovector original
    max_values = {i: abs(autovector[i]) for i in range(len(autovector))}
    max_values2 = {i: abs(autovector2[i]) for i in range(len(autovector2))}
    max_values = dict(sorted(max_values.items(), key=lambda item: item[1], reverse=True))
    max_values2 = dict(sorted(max_values2.items(), key=lambda item: item[1], reverse=True))
    return max_values,max_values2

# def find_max_AVE_position(autovector):
#     max_value = autovector[0]
#     position = 0
#     for i in range(len(autovector)):
#         if autovector[i] > max_value:
#             max_value = autovector[i]
#             position = i
#     return position, max_value

# posicion,value = find_max_AVE_position(autovector)
# posicion2,value2 = find_max_AVE_position(autovector2)

max_values,max_values2 = find_max_AVE_position(autovector, autovector2)
max_values = list(max_values.items())[:5]
max_values2 = list(max_values2.items())[:5]

print(max_values)
print(max_values2) 




