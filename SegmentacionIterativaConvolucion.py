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

def FazConvolucion(img,mask,resolution,std,threshold):
    #Kernel X
    conv2=cv2.filter2D(img/255, -1, mascaraxreparada(resolution,std))
    conv3=cv2.filter2D(img/255, -1, mascaracruzreparada(resolution,std))
    #Promedio convoluciones X y +
    conv4=(conv2+conv3)/2 #Preparo la convolucion
    #cv2.imwrite("resolution "+str(resolution)+" std "+str(std)+".png",clip(conv4))
    ret, thresholded=cv2.threshold(conv4,threshold*conv4.max(),255,cv2.THRESH_BINARY_INV) #Todo lo que tenga un valor por encima de threshold*maximo se pone a cero. Lo demas, a 255.
    CalculoError=thresholded+mask #Ahora sumo las dos mascaras. Aquellos píxeles que sumen 255 significa que las máscaras no coinciden (una es blanca y otra, negra). Luego, contando los píxeles que están a 255 en esta matriz, se calcula automaticamente el error sin tener que normalizar a 1 ni nada.
    Error=CalculoError[CalculoError==255].size
    return(thresholded,100*Error/img.size)

def calibracion(img,mask):
    result=np.zeros(4)
    MinError=9999999999
    for threshold in list(np.linspace(0,0.4,41)):
        _,Error=FazConvolucion(img,mask,45,0.177777777,threshold) #Resolucion 45, std=8
        if (Error<MinError):
            MinError=Error
            result=np.array([45,0.177777777,threshold,Error])
    return(list(result))

def calibracioniterativa(names,masks):
    resultados=list()
    for i in range(len(names)):
        img=cv2.imread(names[i],0)
        mask=cv2.imread(masks[i],0)
        ResultadoImagen=calibracion(img,mask) #Calibro la imagen numero 'i'
        ResultadoImagen.append(names[i]) #Añado el nombre
        resultados.append(ResultadoImagen) #Añado el resultado a los demas
        print("Llevo: "+str(i+1)+" Imágenes de "+str(len(names))+". Calibracion para esta imagen: "+str(ResultadoImagen[0])+", "+str(ResultadoImagen[1])+", "+str(ResultadoImagen[2])+", "+str(ResultadoImagen[3]))
    return(resultados)

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

def Segmentacioniterativa(names,masks):
    errores=list()
    for i in range(len(names)):
        img=cv2.imread(names[i],0)
        mask=cv2.imread(masks[i],0)
        MascaraConvolucion,error=FazConvolucion(img,mask,45,8,0.2)
        errores.append(error)
        resultado=superponerFAZ(img, MascaraConvolucion, 0.35)
        cv2.imwrite(names[i]+"_Segmentado.png",resultado)
        print("Llevo: "+str(i+1)+" Imágenes de "+str(len(names)))
    return(errores)

#------------------------
#Calibracion
#------------------------

ImagenesOriginales,Mascaras=search_files(os.getcwd())
Calibracion=calibracioniterativa(ImagenesOriginales,Mascaras) #Al nombrar en el argumento search_files, esta línea es la que borra los plexos 2 y 4.
        
#------------------------
#Programa Principal
#------------------------

# ImagenesOriginales,Mascaras=search_files(os.getcwd())
# Errores=Segmentacioniterativa(ImagenesOriginales,Mascaras) #Al nombrar en el argumento search_files, esta línea es la que borra los plexos 2 y 4.
# plt.hist(Errores,bins=100)