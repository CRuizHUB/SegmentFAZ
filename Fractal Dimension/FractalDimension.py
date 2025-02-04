# -*- coding: utf-8 -*-
"""Segmentación A-Scan en bucle.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19QS6RBkx5gHtxhsxUUeWVzKYtMq6kp68

Primero, cargamos librerías:
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2
import os.path
from os import path
from PIL import Image
from scipy.optimize import curve_fit

"""Defino funciones para cargar y guardar imágenes. También puede usarse cv2, pero algunas funciones usan estas como subrutinas."""

imagen=cv2.imread("a.png",0) #tiene que ser mascara de ceros y unos
sum_rows = np.sum(imagen,axis=0)
sum_cols = np.sum(imagen,axis=1)
iniciocolumnas=0
finalcolumnas=0
iniciofilas=0
finalfilas=0
for i in range(len(sum_rows)-1):
    if(sum_rows[i]==0 and sum_rows[i+1]!=0):
        iniciofilas=i+1
    elif(sum_rows[i]!=0 and sum_rows[i+1]==0):
        finalfilas=i
for i in range(len(sum_cols)-1):
    if(sum_cols[i]==0 and sum_cols[i+1]!=0):
        iniciocolumnas=i+1
    elif(sum_cols[i]!=0 and sum_cols[i+1]==0):
        finalcolumnas=i+1
if (finalcolumnas==0):
    finalcolumnas=imagen.shape[1]
if (finalfilas==0):
    finalfilas=imagen.shape[0]
tamano=[finalfilas-iniciofilas,finalcolumnas-iniciocolumnas]
lado=np.max(tamano)
curva=imagen[iniciocolumnas:iniciocolumnas+lado,iniciofilas:iniciofilas+lado]
#cv2.imwrite("aislada.png",curva) Guarda la curva recortada

perimetro=0
perimetro=curva.sum()/255

dimension=np.log(perimetro)/np.log(lado)
print(f"La dimensión de la curva es:{dimension}")