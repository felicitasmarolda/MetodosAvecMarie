#Ejercicio 2.1

import numpy as np
import matplotlib.pyplot as plt
import funciones_auxiliares as fa
import os
from PIL import Image

#importar las imagenes de la carpeta dataset_img como vectores
#importamos img00.jpeg
file_path = 'datasets_imgs/img00.jpeg'
p = 28

img_vector = fa.cargar_y_transformar_imagen(file_path, p)
if img_vector is not None:
    print(f"Vector de la imagen de tamaño {p}x{p}:")
    print(img_vector)
    print(f"Tamaño del vector: {img_vector.shape}")

