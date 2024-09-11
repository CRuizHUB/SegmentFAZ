# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 10:10:11 2023

@author: villa
"""

import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
import os.path
from scipy.optimize import curve_fit
import seaborn as sns
import pandas as pd

def clip(result):
    result2=result-result.min()
    result2=result2/result2.max()
    return(255*result2)
    

def gaussiana(longitud,avg,std):
    base=range(longitud)
    function=np.zeros_like(base).astype(np.float32)
    for i in range(longitud):
        function[i]=np.exp(-((base[i]-avg)*(base[i]-avg))/(2*std*std))
    return(function)

def mascaracruz(resolution,std):
    mascara=np.zeros((resolution,resolution))
    for i in range(resolution):
        mascara[:,i]=mascara[:,i]+gaussiana(resolution,resolution/2,std)
        mascara[i,:]=mascara[i,:]+gaussiana(resolution,resolution/2,std)
    return(mascara)

def mascarax(resolution,std):
    mascara=np.zeros((resolution,resolution))
    for i in range(resolution):
        mascara[:,i]=mascara[:,i]+gaussiana(resolution,i,std)
        mascara[i,:]=mascara[i,:]+gaussiana(resolution,i,std)
        mascara[:,i]=mascara[:,i]+gaussiana(resolution,resolution-i,std)
        mascara[i,:]=mascara[i,:]+gaussiana(resolution,resolution-i,std)
    return(mascara/2)

def mascaraizder(resolution,std):
    mascara=np.zeros((resolution,resolution))
    for i in range(resolution):
        mascara[:,i]=mascara[:,i]+gaussiana(resolution,i,std)
    return(mascara)

def mascaraderiz(resolution,std):
    mascara=np.zeros((resolution,resolution))
    for i in range(resolution):
        mascara[:,i]=mascara[:,i]+gaussiana(resolution,resolution-i,std)
    return(mascara)

def mascaraxreparada(resolution, std):
    mascarax=np.zeros([resolution,resolution,3]) #Creo una pila de 3 matrices. En la primera meto la línea de izquierda a derecha, y, en la segunda, de derecha a izquierda.
    mascarax[:,:,0]=mascaraizder(resolution,std)
    mascarax[:,:,1]=mascaraderiz(resolution,std)
    mascarax[mascarax<(200/255)]=0 # Si el redondeo es muy agresivo porque borra las colas, bajar a 100/255.
    for i in range(resolution):
        for j in range(resolution):
            if (mascarax[i,j,0]!=0 and mascarax[i,j,1]!=0):
                mascarax[i,j,2]=(mascarax[i,j,0]+mascarax[i,j,1])/4
                
    mascarax[:,:,0]=mascarax[:,:,0]-mascarax[:,:,2]
    mascarax[:,:,1]=mascarax[:,:,1]-mascarax[:,:,2]
    return(mascarax[:,:,0]+mascarax[:,:,1])

def mascaracruzreparada(resolution, std):
    mascara=np.zeros([resolution,resolution,3]) #Creo una pila de 3 matrices. En la primera meto la línea de izquierda a derecha, y, en la segunda, de derecha a izquierda.
    for i in range(resolution):
        mascara[:,i,0]=mascara[:,i,0]+gaussiana(resolution,resolution/2,std) #linea vertical
        mascara[i,:,1]=mascara[i,:,1]+gaussiana(resolution,resolution/2,std) #Linea horizontal
    mascara[mascara<(200/255)]=0 # Si el redondeo es muy agresivo porque borra las colas, bajar a 100/255.
    for i in range(resolution):
        for j in range(resolution):
            if (mascara[i,j,0]!=0 and mascara[i,j,1]!=0):
                mascara[i,j,2]=(mascara[i,j,0]+mascara[i,j,1])/4
                
    mascara[:,:,0]=mascara[:,:,0]-mascara[:,:,2]
    mascara[:,:,1]=mascara[:,:,1]-mascara[:,:,2]
    return(mascara[:,:,0]+mascara[:,:,1])

def mascarainglesa(resolution,std):
    #Mascara cruz
    mascara=np.zeros([resolution,resolution,3]) #Creo una pila de 3 matrices. En la primera meto la línea de izquierda a derecha, y, en la segunda, de derecha a izquierda.
    mascara[:,:,0]=mascaracruzreparada(resolution, std)
    mascara[:,:,1]=mascaraxreparada(resolution, std)
    mascara[mascara<(200/255)]=0 # Si el redondeo es muy agresivo porque borra las colas, bajar a 100/255.
    for i in range(resolution):
        for j in range(resolution):
            if (mascara[i,j,0]!=0 and mascara[i,j,1]!=0):
                mascara[i,j,2]=(mascara[i,j,0]+mascara[i,j,1])/4
                
    mascara[:,:,0]=mascara[:,:,0]-mascara[:,:,2]
    mascara[:,:,1]=mascara[:,:,1]-mascara[:,:,2]
    return(mascara[:,:,0]+mascara[:,:,1])

def FazConvolucion(img,mask,resolution,std,threshold):
    #Kernel X
    conv4=cv2.filter2D(img/255, -1, mascarainglesa(resolution,std))
    ret, thresholded=cv2.threshold(conv4,threshold*conv4.max(),255,cv2.THRESH_BINARY_INV) #Todo lo que tenga un valor por encima de threshold*maximo se pone a cero. Lo demas, a 255.
    CalculoError=thresholded+mask #Ahora sumo las dos mascaras. Aquellos píxeles que sumen 255 significa que las máscaras no coinciden (una es blanca y otra, negra). Luego, contando los píxeles que están a 255 en esta matriz, se calcula automaticamente el error sin tener que normalizar a 1 ni nada.
    Error=CalculoError[CalculoError==255].size
    return(thresholded,100*Error/img.size)

def calibracion(img,mask):
    result=np.zeros(4)
    MinError=9999999999
    for resolution in list([45]):
        for std in list([0.17777777777777777777]):
            for threshold in list(np.linspace(0,0.4,41)):
                _,Error=FazConvolucion(img,mask,resolution,std,threshold)
                if (Error<MinError):
                    MinError=Error
                    result=np.array([resolution,std,threshold,Error])
    return(list(result))

def search_files(path):
    os.chdir(path)
    names=[]
    masks=[]
    files = glob.glob('*.tif')
    for file in files:
        if file[-12] == '1':
            names.append(file)
            #masks.append(file+"_promedio.jpg") #Si solo se queren usar las Mascaras E1, van nombradas distintas. Comentar esta linea y usar la de abajo
            masks.append(file[:-4]+".jpg")
        else:
            os.remove(file) #Si la iamgen no es del plexo superficial, se eliminara. Por eso te recomiendo que ejecutes este .py en una carpeta copia y no en la original.
    return (names,masks)

def superponerFAZ(img,resultadofinal,transparencia): #Esto es un algoritmo aparte que permite superponer una imagen binarizada sobre otra cualquiera. Primero se introduce la imagen de fondo, luego la binarizada, y por último, la transparencia. Si es 1, la imagen binarizada no se ve. Si es 0, solo se ve la imagen binarizada.
  A=img.copy()
  resultadofinalcolor=np.zeros((A.shape[0],A.shape[1],3))
  resultadofinalcolor[:,:,2]=resultadofinal #Convierte los blancos de una imagen binarizada en rojos (Asigno la imagen binarizada al canal rojo, y dejo a 0 los otros dos).
  for i in range(resultadofinalcolor.shape[0]):
    for j in range(resultadofinalcolor.shape[1]):
      for k in range(resultadofinalcolor.shape[2]):
        if (resultadofinalcolor[i,j,k]<128): #En las zonas donde la imagen binarizada sea negra, coloca la imagen de fondo. El usar como threshold 128 permite eliminar un pequeño artefacto en la imagen que emborrona el area próxima a la FAZ. ¿Puede ser por la compresión?
          resultadofinalcolor[i,j,k]=A[i,j] #No hay indice K porque en blanco y negro, los tres indices son iguales.
        else:
          resultadofinalcolor[i,j,k]=transparencia*A[i,j]+(1-transparencia)*resultadofinalcolor[i,j,k] #Si estoy en un punto 'rojo', mezclo información de ambas imágenes dependiendo de la transparencia deseada.
  return(resultadofinalcolor)

def superponerdobleFAZ(img,resultadofinal,resultadofinal2,transparencia): #Esto es un algoritmo aparte que permite superponer una imagen binarizada sobre otra cualquiera. Primero se introduce la imagen de fondo, luego la binarizada, y por último, la transparencia. Si es 1, la imagen binarizada no se ve. Si es 0, solo se ve la imagen binarizada.
  A=img.copy()
  resultadofinalcolor=np.zeros((A.shape[0],A.shape[1],3))
  resultadofinalcolor[:,:,2]=resultadofinal #Convierte los blancos de una imagen binarizada en rojos (Asigno la imagen binarizada al canal rojo, y dejo a 0 los otros dos).
  resultadofinalcolor[:,:,1]=resultadofinal2
  for i in range(resultadofinalcolor.shape[0]):
    for j in range(resultadofinalcolor.shape[1]):
      for k in range(resultadofinalcolor.shape[2]):
        if (resultadofinalcolor[i,j,k]<128): #En las zonas donde la imagen binarizada sea negra, coloca la imagen de fondo. El usar como threshold 128 permite eliminar un pequeño artefacto en la imagen que emborrona el area próxima a la FAZ. ¿Puede ser por la compresión?
          resultadofinalcolor[i,j,k]=A[i,j] #No hay indice K porque en blanco y negro, los tres indices son iguales.
        else:
          resultadofinalcolor[i,j,k]=transparencia*A[i,j]+(1-transparencia)*resultadofinalcolor[i,j,k] #Si estoy en un punto 'rojo', mezclo información de ambas imágenes dependiendo de la transparencia deseada.
  return(resultadofinalcolor)

def Segmentacioniterativa(names,masks):
    errores=list()
    areasexternas=list()
    areasinternas=list()
    erroresexternos=list()
    acircularidadesinternas=list()
    acircularidadesexternas=list()
    perimetrosexternos=list()
    perimetrosinternos=list()
    ratiosinternos=list()
    ratiosexternos=list()
    for i in range(len(names)):
        img=cv2.imread(names[i],0)
        mask=cv2.imread(masks[i],0)
        MascaraConvolucion,error=FazConvolucion(img,mask,45,8,0.2)
        MascaraConvolucionExterna,error2=FazConvolucion(img,mask,45,8,0.25)
        errores.append(error)
        erroresexternos.append(error2)
        #Calculo todo
        perimetrointerno,areainterno,acircularidadinterno,ratiointerno,_,_=statsFAZ(MascaraConvolucion)
        perimetroexterno,areaexterno,acircularidadexterno,ratioexterno,_,_=statsFAZ(MascaraConvolucionExterna)
        areasexternas.append(MascaraConvolucionExterna.sum()*9/(102400*255))
        areasinternas.append(MascaraConvolucion.sum()*9/(102400*255))
        acircularidadesinternas.append(acircularidadinterno)
        acircularidadesexternas.append(acircularidadexterno)
        perimetrosexternos.append(perimetroexterno*3/320)
        perimetrosinternos.append(perimetrointerno*3/320)
        ratiosinternos.append(ratiointerno)
        ratiosexternos.append(ratioexterno)
        #Dibujo resultado
        resultado=superponerdobleFAZ(img, MascaraConvolucion, MascaraConvolucionExterna, 0.35)
        cv2.imwrite(names[i]+"_Segmentado.png",resultado)
        print("Llevo: "+str(i+1)+" Imágenes de "+str(len(names)))
    return(perimetrosinternos,perimetrosexternos,areasinternas,areasexternas,acircularidadesinternas,acircularidadesexternas,ratiosinternos,ratiosexternos,errores,erroresexternos)
        
#Aquí van las funciones que necesito para convertir una máscara binaria en una lista de coordenadas y sacar la dimension fractal y lo demas.

def vertical(mascarabinaria):
    contornomascara=np.zeros_like(mascarabinaria)
    for j in range(mascarabinaria.shape[1]):
        contornomascara[:,j]=np.append(np.diff(mascarabinaria[:,j]),0)
    contornomascara[contornomascara!=0]=255
    return(contornomascara)

def horizontal(mascarabinaria):
    contornomascara=np.zeros_like(mascarabinaria)
    for i in range(mascarabinaria.shape[0]):
        contornomascara[i,:]=np.append(np.diff(mascarabinaria[i,:]),0)
    contornomascara[contornomascara!=0]=255
    return(contornomascara)

def sacaborde(imagen): #Me baso en que la derivada de una mascara binaria es distinta de 0 solo en el borde.
    return(vertical(imagen)+horizontal(imagen))

def distancia(punto1,punto2): #Punto1 y punto2 son dos arrays de longitud 2
  return(np.sqrt((punto2[0]-punto1[0])*(punto2[0]-punto1[0])+(punto2[1]-punto1[1])*(punto2[1]-punto1[1])))

def func(x,a,b,direccion): #Esto es una elipse en polares centrada en el origen, con el eje mayor inclinado un ángulo 'direccion'. A y B son los SEMIEJES, luego, a la hora de dar el resultado, hay que multiplicarlos por dos. Theta está escrita como 'x' para que lo detecte el algoritmo de ajuste.
    return(np.power((np.sin(x-direccion)/a)*(np.sin(x-direccion)/a)+(np.cos(x-direccion)/b)*(np.cos(x-direccion)/b),-1/2))

def statsFAZ(mascarabinaria): # ContornoFAZ es una lista de arrays de dos dimensiones en cartesianas.
  if(mascarabinaria.sum()==0):
      return(np.zeros(6))
  perimetro=0
  sumax=0
  sumay=0
  area=0
  contornofaz = list(np.where(sacaborde(mascarabinaria) == 255))
  contornofaz=list(np.transpose(contornofaz))
  for i in range(len(contornofaz)):
      perimetro=perimetro+distancia(contornofaz[i-1],contornofaz[i]) #El primer dato calculado es la distancia entre el punto -1 (último) y el 0 (primero), de forma que al sumar la última distancia entre el penúltimo y el último punto, se cierra el contorno.
      area=area+(contornofaz[i-1][0]*contornofaz[i][1]-contornofaz[i][0]*contornofaz[i-1][1])/2 #Método Shoelacer
      sumax=sumax+contornofaz[i][0] #Sumo las coordenadas X e Y para calcular el centro.
      sumay=sumay+contornofaz[i][1]
  centrofaz=np.array([(sumax/len(contornofaz)),(sumay/len(contornofaz))])
  contornopolaresR=np.zeros(len(contornofaz))
  contornopolaresTHETA=np.zeros(len(contornofaz)) #Me interesa expresar R y THETA como dos arrays separados porque para calcular la acircularidad y los ajustes, es mejor tener dos arrays, como si fuese una función y(x).
  contornocentrado=list(contornofaz) #copio el contorno que he recibido como argumento
  for i in range(len(contornofaz)):
      contornocentrado[i]=contornofaz[i]-centrofaz #y a cada punto le resto las coordenadas del centro. Como resultado, lo que tengo es la misma curva, pero centrada en el origen, de forma que ya puedo pasar a polares.
      contornopolaresR[i]=distancia(contornocentrado[i],np.array([0,0])) #La coordenada radial es la distancia al origen
      contornopolaresTHETA[i]=np.arctan(contornocentrado[i][1]/contornocentrado[i][0]) #La coordenada theta es la arcotangente del cociente de las coordenadas cartesianas y/x
  #Ahora que ya tengo la curva en polares, calculo la acircularidad, los ejes, y la orientación del eje mayor
  acircularidad=np.std(contornopolaresR)/np.average(contornopolaresR) #La acircularidad es la desviación estándar de la coordenada radial.
  popt, pcov = curve_fit(func, contornopolaresTHETA, contornopolaresR) #En popt aparecerán los valores ajustados de a,b y dirección en ese orden.
  return(np.array([perimetro,area,acircularidad,max((popt[0]/popt[1]),(popt[1]/popt[0])),2*popt[0],popt[2]])) #Devuelve los parámetros en el siguiente orden: Perímetro, área, acircularidad, ratio, eje mayor y dirección.
  
# Fin de cálculos de la FAZ.

def create_dataframe(x_coordinates, y_coordinates,TituloX,TituloY): #Le pasas dos listas de numeros, y te crea un dataframe para plotear en Seaborn. Luego, tienes que escribir el nombre con el que quieres que salgan los ejes X e Y
    # Create a DataFrame from X and Y coordinates
    data = {TituloX: x_coordinates, TituloY: y_coordinates}
    df = pd.DataFrame(data)
    return df

#------------------------
#Programa Principal
#------------------------

ImagenesOriginales,Mascaras=search_files(os.getcwd())
perimetrosinternos,perimetrosexternos,areasinternas,areasexternas,acircularidadesinternas,acircularidadesexternas,ratiosinternos,ratiosexternos,erroresinternos,erroresexternos=Segmentacioniterativa(ImagenesOriginales,Mascaras) #Al nombrar en el argumento search_files, esta línea es la que borra los plexos 2 y 4.