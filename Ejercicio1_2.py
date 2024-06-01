#Ejercicio 1.2
 # agarramos de las d dimanesiones mas importantes agarramos sus autovetores (lo que vienen de los valore ssingulares) y nos fijamos el numero mas grande de todos, nos fijamos en que posicion de su vector esta y esa posicion va a ser la dimension que mass dice

import numpy as np
import matplotlib.pyplot as plt
import os
import funciones_auxiliares as fa

#Cargar los datos del ejercicio 1_1
from Ejercicio1_1 import X
from Ejercicio1_1 import Y

#Hacer SVD de X
U, S, Vt = np.linalg.svd(X, full_matrices=False)

#buscamos los 2 autovectores más importantes
autovector = Vt[0]
autovector2 = Vt[1]

#buscamos el valor más grande de cada autovector y su posicion en el autovector
def find_max_AVE_position(autovector):
    max_value = autovector[0]
    position = 0
    for i in range(len(autovector)):
        if autovector[i] > max_value:
            max_value = autovector[i]
            position = i
    return position, max_value

posicion,value = find_max_AVE_position(autovector)
posicion2,value2 = find_max_AVE_position(autovector2)

if(value>value2):
    posicion = posicion
else:
    posicion = posicion2
print("La dimensión más importante es: ", posicion)



