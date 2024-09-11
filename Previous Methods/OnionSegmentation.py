# -*- coding: utf-8 -*-
"""
Created on Mon May  8 11:51:44 2023

@author: villa
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2
from PIL import Image
import os
from scipy.optimize import curve_fit
import math

"""Defino funciones para cargar y guardar imágenes. También puede usarse cv2, pero algunas funciones usan estas como subrutinas."""

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def save_image( npdata, outfilename ) :
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
    img.save( outfilename )

"""Primero detectamos los bordes de una imagen de entrada. Los parametros son, en primer lugar, una imagen en blanco y negro, y después los parámetros de anchura y altura del kernel para realizar el TopHat."""

def detectabordes(imgoriginal,anchuraKernel,alturaKernel):
  img=cv2.GaussianBlur(imgoriginal,(5,7),0) #Estos valores son para el blur previo al TopHat. No varía mucho de unos valores a otros.
  rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (anchuraKernel, alturaKernel)) #Los valores buenos para la prueba son 17,19
  tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, rectKernel)
  img2=cv2.Canny(tophat, threshold1=0, threshold2=255)
  return(img2)

"""Ahora, voy a definir dos funciones. En la primera, se analiza un array de entrada y se busca la mayor cadena de ceros consecutivos, y se devuelve un array con LAS POSICIONES en el array donde se encuentra esa cadena de ceros.

La segunda es una implementación en bucle donde, tras introducir una matriz, se aplica la función anterior a cada una de sus filas y se devuelve una lista de arrays, donde la posición 'j' de la lista contiene el array resultante de ejecutar la función anterior con la fila 'j' de la matriz (o imagen, en nuestro caso).
"""

def posicionesarray(entrada):
  listalarga=[] #Creo dos listas. Una contiene la cadena actual de ceros, y la otra, la cadena 'record', la más larga.
  listaactual=[]
  for i in range(len(entrada)): #Para cada elemento del array de entrada...
    if (entrada[i]==0): #Comprueba primero si el valor es 0.
      listaactual.append(i) #En caso afirmativo, cojes la posicion donde esta ese 0, y se la añades a la lista donde estás colocando las posiciones de los ceros.
    else: #En caso contrario...
      if(len(listaactual)>len(listalarga)): #Deja 'listaactual' como está y compara su longitud con la de la cadena 'record' (que será 0 al inicio)
        listalarga=list(listaactual) #Si la cadena de ceros que estaba analizando es más larga que la record, se convierte en la cadena más larga. Si no, la ignoro.
      listaactual=[] #Como este elemento ya no es 0, la cadena actual de ceros termina y la borro, y empezaré a acumular las posiciones de una nueva cadena cuando vuelva a encontrar ceros.
  return(np.array(listalarga)) #Devuelvo el resultado. Como listalarga era una lista, la convierto para que el resultado del algoritmo sea un array de numpy.

def segmentamatriz(matriz):
  posicionesamarcar=[] #Mi intención ahora es acumular los resultados obtenidos para cada fila de la imagen, que son arrays en sí mismas. Creo una lista donde iré anexionando los arrays que me devuelva el algoritmo anterior fila por fila.
  for k in range(matriz.shape[0]): #Para cada fila de la imagen...
    posicionesamarcar.append(posicionesarray(np.ravel(matriz[k,:]))) #Básicamente lo que hago es añadir al final el resultado que sale de analizar la última fila que he procesado. np.ravel 'saca' de la matriz la fila, de forma que no haya problemas a la hora de la lectura. Es una buena práctica, nada más.
  return(posicionesamarcar) #Devuelvo la lista de arrays una vez que he acabado

"""Ahora, se definen las funciones que segmentan la imagen Canny y que filtran por longitudes."""

def segmentafilas(img2,thresholdx,thresholdY): #Esto devuelve la máscara filtrada, pero sin blur. La imagen de entrada es la imagen con detección de bordes canny.
  pixelesapintar=segmentamatriz(img2)

  MascaraROI=np.zeros_like(img2) #Inicializo la máscara.

  for i in range(len(pixelesapintar)):
    for j in range(pixelesapintar[i].size):
      MascaraROI[i,int(pixelesapintar[i][j])]=255

  pixelesblancosenfila=[]

  for i in range(MascaraROI.shape[1]):
    pixelesblancosenfila.append(pixelesapintar[i].size)

  for i in range(MascaraROI.shape[0]):
      if (pixelesblancosenfila[i]<thresholdx*max(pixelesblancosenfila)): 
        MascaraROI[i,:]=np.zeros_like(MascaraROI[i,:]) 

  sumaY=[]

  for i in range(MascaraROI.shape[1]):
    sumaY.append(np.sum(np.ravel(MascaraROI[:,i])))

  for i in range(MascaraROI.shape[1]):
      if (sumaY[i]<thresholdY*max(sumaY)):
        MascaraROI[:,i]=np.zeros_like(MascaraROI[:,i])

  return(MascaraROI)

def segmentacolumnas(img2,thresholdx,thresholdY): #Llamo al argumento img2 porque ese es el nombre que le dí a la imagen inicial en el codigo de arriba.
  A=segmentafilas(img2.transpose(),thresholdx,thresholdY)
  return(A.transpose()) #Lo dicho. Llamo al algoritmo anterior con los mismos parámetros de entrada pero con la matriz inicial transpuesta. Y el resultado que obtengo, lo transpongo de nuevo, y eso es lo que devuelvo.

"""A continuación, en esta función, se introduce el resultado de una segmentación y rodea el mayor perímetro. Muestra una imagen pero no devuelve ningun valor. A veces no cierra el perímetro. Primero se introduce la imagen binaria, luego la imagen de fondo."""

def detectaperimetro(MascaraROI,img):
  cnts,hierarchy = cv2.findContours(MascaraROI.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  print(str(len(cnts))+' contours detected')

  area = np.array([cv2.contourArea(cnts[i]) for i in range(len(cnts))]) # list of all areas
  maxa_ind = np.argmax(area) # index of maximum area contour

  xx = [cnts[maxa_ind][i][0][0] for i in range(len(cnts[maxa_ind]))]
  yy = [cnts[maxa_ind][i][0][1] for i in range(len(cnts[maxa_ind]))]
  plt.imshow(img, cmap='gray')
  plt.plot(xx,yy,'r',linewidth=1)
  plt.title('RESULTADO')

"""Estas funciones definen una tercera imagen binaria en funcion de otras dos, aplicando a cada píxel la operación lógica OR y AND."""

def mascaraOR(mascara1,mascara2):
    resultado = np.zeros_like(mascara1)
    resultado[(mascara1 == 255) | (mascara2 == 255)] = 255
    return resultado

def mascaraXOR(mascara1, mascara2):
    resultado = np.zeros_like(mascara1)
    resultado[mascara1 != mascara2] = 255
    return resultado

def mascaraAND(mascara1, mascara2):
    resultado = np.zeros_like(mascara1)
    resultado[(mascara1 == 255) & (mascara2 == 255)] = 255
    return resultado

"""Ahora, voy a definir un algoritmo que, dada la imagen de detección de bordes Canny, devuelve la imagen rotada sobre un fondo blanco:"""

def rotarimagen(img2,angulo): #El angulo va en grados
  fondo=np.full((int(img2.shape[0]*1.4143),int(img2.shape[1]*1.4143)),255) #Crea una matriz llena de '255' de tamaño sqrt(2)*imagen_original
  x_offset=y_offset=int(img2.shape[0]*0.20710678118) #offset es la distancia entre la esquina superior derecha entre las dos imágenes. Se puede demostrar con trigonometria que, para que las imágenes queden concentricas, la distancia entre los vértices ha de ser (1-(1/sqrt(2)))*tamaño imagen original. Como el offset se descompone en X y en Y a partes iguales, sera offsetx=offsety=(1-(1/sqrt(2)))/sqrt(2)=0.20710678118*(tamano original)
  fondo[int(y_offset):int(y_offset+np.ravel(img2.shape[0])), int(x_offset):int(x_offset+np.ravel(img2.shape[1]))] = img2
  fondo=np.full_like(fondo,255)-fondo #Invierto colores. Ahora la imagen es blanca con lineas negras sobre un fondo negro
  rotate_matrix = cv2.getRotationMatrix2D((int(fondo.shape[0]/2),int(fondo.shape[1]/2)),angulo,scale=1) #ESTO NO ES LA IMAGEN ROTADA. Es la matriz de la transformación
  height, width = fondo.shape[:2]
  save_image(fondo,"fondo.jpg") #Por alguna razon, si intentas pasar la matriz 'fondo' sin guardar como imagen y volver a cargar, da error.
  rotadainvertida = cv2.warpAffine(cv2.imread("fondo.jpg",0), rotate_matrix, dsize=(width, height)) # ESTA SI es la linea que rota la imagen
  rotada=np.full_like(rotadainvertida,255)-rotadainvertida #Ahora, vuelvo a invertir la imagen para que quede como estaba antes, pero con todo el lienzo blanco.
  for i in range(rotada.shape[0]):
    for j in range(rotada.shape[1]):
      if (rotada[i,j]<128):
        rotada[i,j]=0
      else:
        rotada[i,j]=255
  os.remove("fondo.jpg")
  return(rotada)

def encuadrar(img2): #Esto centra la imagen con un marco blanco alrededor. Sirve para que al final, el contorno y la FAZ de '1.jpg' sean concéntricas.
  angulo=0
  fondo=np.full((int(img2.shape[0]*1.4143),int(img2.shape[1]*1.4143)),255) #Crea una matriz llena de '255' de tamaño sqrt(2)*imagen_original
  x_offset=y_offset=int(img2.shape[0]*0.20710678118) #offset es la distancia entre la esquina superior derecha entre las dos imágenes. Se puede demostrar con trigonometria que, para que las imágenes queden concentricas, la distancia entre los vértices ha de ser (1-(1/sqrt(2)))*tamaño imagen original. Como el offset se descompone en X y en Y a partes iguales, sera offsetx=offsety=(1-(1/sqrt(2)))/sqrt(2)=0.20710678118*(tamano original)
  fondo[int(y_offset):int(y_offset+np.ravel(img2.shape[0])), int(x_offset):int(x_offset+np.ravel(img2.shape[1]))] = img2
  fondo=np.full_like(fondo,255)-fondo #Invierto colores. Ahora la imagen es blanca con lineas negras sobre un fondo negro
  rotate_matrix = cv2.getRotationMatrix2D((int(fondo.shape[0]/2),int(fondo.shape[1]/2)),angulo,scale=1) #ESTO NO ES LA IMAGEN ROTADA. Es la matriz de la transformación
  height, width = fondo.shape[:2]
  save_image(fondo,"fondo.jpg") #Por alguna razon, si intentas pasar la matriz 'fondo' sin guardar como imagen y volver a cargar, da error.
  rotadainvertida = cv2.warpAffine(cv2.imread("fondo.jpg",0), rotate_matrix, dsize=(width, height)) # ESTA SI es la linea que rota la imagen
  rotada=np.full_like(rotadainvertida,255)-rotadainvertida #Ahora, vuelvo a invertir la imagen para que quede como estaba antes, pero con todo el lienzo blanco.
  os.remove("fondo.jpg")
  return(rotada)

def quitamarco(imagen): #Esta función, dada una imagen con marco blanco, quita el marco blanco. La imagen a sacar ha de ser cuadrada y estar dentro de un marco cuadrado.
  x_offset=y_offset=int((imagen.shape[0]/1.4143)*0.20710678118) #Podemos expresar todas las distancias en función del tamaño de la imagen enmarcada porque asumimos (y ocurre) que la imagen dentro del marco está centrada.
  return(imagen[int(y_offset):int(-y_offset+np.ravel(imagen.shape[0])), int(x_offset):int(-x_offset+np.ravel(imagen.shape[1]))])

"""Ahora, lo que haremos será realizar un escaneo horizontal para diferentes ángulos, y calcularemos la máscara AND de todas las que obtengamos para diferentes ángulos. Se devolverá esa máscara AND. Los argumentos se introducen en este orden:


1.   Imagen a analizar
2.   Tamaño en grados del paso. Se escanea de 0 a 180 grados
3.   i,j,k,l son los thresholds para las segmentaciones en fila y columna
4.   ThresholdZ es un parámetro similar a ThresholdX y ThresholdY. Decide si en la máscara final se dibuja blanco o negro. 


"""

def segmentacionrotatoria(img2,tamanopaso,i,j,k,l,thresholdZ):
  ROIROTADAS=np.zeros((int(img2.shape[0]*1.4143),int(img2.shape[1]*1.4143),len(range(0,180,tamanopaso)))) #Creo una matriz 3D, donde cada 'plano' es una MascaraROI.
  paso=0

  for i in range(0,180,tamanopaso):
    imagenactual=rotarimagen(img2,i) #Roto la imagen Canny 'i' grados
    MascaraROI=mascaraOR(segmentafilas(imagenactual,i,j),segmentacolumnas(imagenactual,k,l)) #Segmenta las filas y las columnas de la imagen obtenida, y calcula la máscara OR.
    rotate_matrix = cv2.getRotationMatrix2D((int(MascaraROI.shape[0]/2),int(MascaraROI.shape[1]/2)),-i,scale=1) #Ahora, roto el resultado para que vuelva a quedar horizontal. (Roto '-i' grados)
    height, width = MascaraROI.shape[:2]
    save_image(MascaraROI,"mascara.jpg") #Por alguna razon, si intentas pasar la matriz 'fondo' sin guardar como imagen y volver a cargar, da error.
    ROIROTADAS[:,:,paso] = cv2.warpAffine(cv2.imread("mascara.jpg",0), rotate_matrix, dsize=(width, height)) # ESTA SI es la linea que rota la imagen
    paso=paso+1

  #Ahora, aplicamos una máscara combinada AND y OR a las matrices 'apelmazadas' en el eje 'z' de ROIROTADAS. Lo que haré será dibujar en una nueva máscara un 0 o un 255 según haya o no un número de píxeles blancos en una cierta posición de ROIROTADAS:

  MascaraFinal=np.sum(ROIROTADAS,axis=2) #Sumo todas las matrices que tengo en ROIROTADAS, séase, 'integro' en el eje Z.
  sumamaxima=np.max(MascaraFinal)
  resultado=np.zeros_like(MascaraFinal) #Creo una máscara, donde escribiré el resultado. Comienza toda a 0.

  for i in range(MascaraFinal.shape[0]):
    for j in range(MascaraFinal.shape[1]):
      if (MascaraFinal[i,j]>thresholdZ*sumamaxima): #Si la suma de ese pixel para cada una de las matrices en ROIROTADAS es superior a la mitad (si ThresholdZ=0.5) de la suma máxima, pinto el pixel de blanco.
        resultado[i,j]=255
  os.remove("mascara.jpg")
  return(quitamarco(resultado))

"""Este es otro algoritmo para pintar el contorno sobre '1.jpg'. Usar si no se detecta bien el perímetro que se muestra justo arriba."""

def superponerFAZ(img,resultadofinal,transparencia): #Esto es un algoritmo aparte que permite superponer una imagen binarizada sobre otra cualquiera. Primero se introduce la imagen de fondo, luego la binarizada, y por último, la transparencia. Si es 1, la imagen binarizada no se ve. Si es 0, solo se ve la imagen binarizada.
  A=img.copy()
  M = cv2.moments(resultadofinal)
  cY = int(M["m10"] / M["m00"])
  cX = int(M["m01"] / M["m00"])
  resultadofinalcolor=np.zeros((A.shape[0],A.shape[1],3))
  resultadofinalcolor[:,:,2]=resultadofinal #Convierte los blancos de una imagen binarizada en rojos (Asigno la imagen binarizada al canal rojo, y dejo a 0 los otros dos).
  for i in range(resultadofinalcolor.shape[0]):
    for j in range(resultadofinalcolor.shape[1]):
      for k in range(resultadofinalcolor.shape[2]):
        if (resultadofinalcolor[i,j,k]<128): #En las zonas donde la imagen binarizada sea negra, coloca la imagen de fondo. El usar como threshold 128 permite eliminar un pequeño artefacto en la imagen que emborrona el area próxima a la FAZ. ¿Puede ser por la compresión?
          resultadofinalcolor[i,j,k]=A[i,j] #No hay indice K porque en blanco y negro, los tres indices son iguales.
        else:
          resultadofinalcolor[i,j,k]=transparencia*A[i,j]+(1-transparencia)*resultadofinalcolor[i,j,k] #Si estoy en un punto 'rojo', mezclo información de ambas imágenes dependiendo de la transparencia deseada.
  resultadofinalcolor[cX,cY,:]=[0,255,0] #Marco el pixel central de verde
  return(resultadofinalcolor)

"""Ahora, definiremos un algoritmo que detecta la FAZ usando el filtrado de detección de centros. La función principal es la última (detectaFAZ)."""

def encuentrazonaplana(centros,threshold):
  posicionesplanas=np.ones_like(centros) #Creo un nuevo array con unos, del mismo tamaño que el array a analizar (centros).
  valorinicial=999999 #De esta forma, siempre se cojerá como valor de referencia el primer elemento del array.
  valoractual=0
  for i in range(centros.size):
    valoractual=centros[i] #Cojemos el valor en la posición i
    if ((valoractual>(1-(threshold/2))*valorinicial) and (valoractual<(1+(threshold/2))*valorinicial)): #Si el valor en la posición 'i' se encuentra un 15% arriba o abajo del valor de referencia...
      posicionesplanas[i]=0 #Asigno un valor 0 a esa posición.
    else: #Cuando encuentre un pico/valle que se desvíe mas del 15%...
      valorinicial=valoractual #Significa que la zona plana ha acabado. Cojo este valor como nueva referencia, y comienzo a escanear de nuevo, buscando otra zona plana.
  #Ahora tendre un array donde, en las posiciones planas es 0, y en el resto, 1. Por lo tanto, puedo usar uno de los algoritmos anteriores que extraía las posiciones de la cadena más larga de ceros encontrada.
  zonaplana=posicionesarray(posicionesplanas) #Y en zonaplana, tendremos las posiciones pertenecientes a la zona plana más larga dentro del array 'centros'.
  return(zonaplana)


def segmentafilassinfiltro(img2): #Esto segmenta la imagen Canny por filas, pero no realiza thresholds.
  pixelesapintar=segmentamatriz(img2)
  MascaraROI=np.zeros_like(img2) #Inicializo la máscara.
  for i in range(len(pixelesapintar)):
    for j in range(pixelesapintar[i].size):
      MascaraROI[i,int(pixelesapintar[i][j])]=255
  return(MascaraROI)



def segmentacolumnassinfiltro(img2): #Llamo al argumento img2 porque ese es el nombre que le dí a la imagen inicial en el codigo de arriba.
  A=segmentafilassinfiltro(img2.transpose())
  return(A.transpose()) #Lo dicho. Llamo al algoritmo anterior con los mismos parámetros de entrada pero con la matriz inicial transpuesta. Y el resultado que obtengo, lo transpongo de nuevo, y eso es lo que devuelvo.



def detectaFAZ(img2,thresholdX,thresholdY): #Aquí, introducimos la imagen tras detectar bodes con CANNY en primer lugar, y luego los thresholds para las filas y columnas.
  pixelesapintar=segmentamatriz(img2)
  MascaraROI=np.zeros_like(img2) #Inicializo la máscara.
  for i in range(len(pixelesapintar)):
    for j in range(pixelesapintar[i].size):
      MascaraROI[i,int(pixelesapintar[i][j])]=255

  centrocadenasblancas=[]

  for i in range(MascaraROI.shape[1]):
    centrocadenasblancas.append(np.average(pixelesapintar[i])) #En esta nueva lista, en el elemento 'j' tendré la posición donde se encuentra el centro de la cadena marcada de blanco. La lista contiene la posición central para cada fila de la imagen.

  centros=np.array(centrocadenasblancas)
  zonaplanahorizontal=encuentrazonaplana(centros,thresholdX)

  ###########

  pixelesapintar=segmentamatriz(img2.transpose())
  MascaraROI=np.zeros_like(img2) #Inicializo la máscara.
  for i in range(len(pixelesapintar)):
    for j in range(pixelesapintar[i].size):
      MascaraROI[i,int(pixelesapintar[i][j])]=255

  centrocadenasblancas=[]

  for i in range(MascaraROI.shape[1]):
    centrocadenasblancas.append(np.average(pixelesapintar[i])) #En esta nueva lista, en el elemento 'j' tendré la posición donde se encuentra el centro de la cadena marcada de blanco. La lista contiene la posición central para cada fila de la imagen.
  centros=np.array(centrocadenasblancas)
  zonaplanavertical=encuentrazonaplana(centros,thresholdY) # Esto encuentra las columnas donde está la FAZ.

  limitesFAZ=[np.min(zonaplanahorizontal),np.max(zonaplanahorizontal),np.min(zonaplanavertical),np.max(zonaplanavertical)] #Introduzco en el array los mínimos y los máximos del rango donde se encuentra la FAZ, en X y en Y. Es, por tanto, un array con 4 valores.
  ROIFilas=segmentafilassinfiltro(img2) #Genero la imagen segmentada por filas
  ROIFilas[0:limitesFAZ[0],:]=0 #Las filas por encina les asigno 0
  ROIFilas[limitesFAZ[1]:ROIFilas.shape[0],:]=0 #Y las que estén por debajo, también
  ROIColumnas=segmentacolumnassinfiltro(img2) #Ahora, repito lo mismo para las columnas
  ROIColumnas[:,0:limitesFAZ[2]]=0
  ROIColumnas[:,limitesFAZ[3]:ROIFilas.shape[1]]=0
  return(mascaraOR(ROIFilas,ROIColumnas)) #Devuelvo la Máscara 'OR' de las dos imagenes anteriores (solo es negro si en las dos imágenes anteriores es negro)

def contornoamano(img): #img ha de leerse en color!
  borde = img[:, :, 1] - img[:, :, 0]
  ret,borde = cv2.threshold(borde,127,255,cv2.THRESH_BINARY)
  #Rellenamos el área
  cnt = cv2.findContours(borde, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
  area = np.zeros(img.shape[:2], np.uint8)
  cv2.drawContours(area, cnt, -1, 255, -1)
  return(area)

def contabilizadiferencias(mascara1,mascara2):
  diferencias=mascaraXOR(mascara1,mascara2) #XOR es el operador que devuelva 0 si los dos bits son iguales o 1 si son distintos.
  suma=np.sum(diferencias) #Voy sumando pixel a pixel. Siempre sumaré 0 o 255
  return(diferencias,100*(suma/(255*diferencias.size))) #Devuelvo 2 argumentos. Por un lado, la máscara en sí, y por otro lado, un numero que es la suma de todos los pixeles dividido entre 255*(Nº píxeles). Sale 0 si la máscara diferencias es negra (imagenes idénticas) y 1 si es blanca (completamente distintas)

def desviacioncuadratica(mascara1,mascara2): #Primero la experimental, luego la teorica. A veces, si la diferencia entre ambos es muy amplia, puede dar lugar a overflow. El resultado sale, pero se añade una advertencia en la salida.
  return((np.sum(mascaraXOR(mascara1,mascara2))/(np.sum(mascara2)))) #Calculo el área de la máscara XOR (Diferencia) y la divido entre el área de la máscara 2. Como los términos de los sumatorios siguen siendo o 0 o 255, no es necesario normalizar a 1 los valores que son 255.

def keep_largest(binary_image,n): #Dada una mascara, devuelve otra con solo las n mayores areas.    
    binary_image = binary_image.astype('uint8')

    # Find contours in the binary image
    contours,_ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Sort contours by area in descending order
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Create a new binary image with the same size as the input image
    result = np.zeros_like(binary_image)
    
    # Draw the two largest contours on the result image
    cv2.drawContours(result, sorted_contours[:n], -1, 255, -1)
    
    return result #No funciona y devuelve error, no soy capaz de averiguar porqué. A ver si va a ser mi ordenador

def une_isletas(img): #Este subalgoritmo tiende a unir varias isletas en caso de que la zona roja superpuesta se vea muy fragmentada.
    result = np.zeros_like(img) #Inicializo una matriz con ceros
    left_fill = np.cumprod(img == 0, axis=1).astype(bool) #Aquí lo que se hace es, por cada fila, ir escribiendo TRUE si todos los elementos anteriores son 0, y false en cuanto haya un elemento distinto de 0, hasta el final de la fila
    right_fill = np.cumprod(np.flip(img == 0, axis=1), axis=1).astype(bool) #Repito el proceso desde el final de la imagen y el sentido opuesto. Para ello, volteo la imagen OJO, NO ES TRANSPONER, con np.flip y hago lo mismo que antes. Ojo, que luego habra que leer este resultado del revés.
    result[left_fill | right_fill[:,::-1]] = 255 #Ahora, lo que hago es, leyendo left_fill del derecho y las filas de right_fill del revés, allí donde uno de los dos píxeles sea blanco, pinto blanco en el resultado final.
    return (-1 * result) + 255 #Cuidado, que lo que se obtiene es lo inverso de lo que queremos. Osea, sale blanco todo menos la faz. Invirtiendo el resultado y sumando 255, se obtiene el resultado final.

def build_histogram(matrix): #calchist no me vale porque solo admite enteros, y NaN es ESTRICTAMENTE un float. No se puede convertir.
    # Flatten the matrix
    pixel_values = matrix.flatten()
    pixel_values = pixel_values[~np.isnan(pixel_values)] #Elimina de la matriz los valores NaN
    # Create an array of zeros to store the histogram values
    hist = np.zeros(256)
    
    # Loop through the pixel values and increment the corresponding histogram bin
    
    for value in pixel_values:
        hist[int(value)] += 1
    return hist

def give_position(inputarray, threshold): 
    max_value = inputarray.max() #cojo el valor maximo
    result = np.where(inputarray > threshold * max_value)[0] #Meto en 'result' las posiciones de los valores que sean mayores a threshold*maximo.
    return result[-1] if len(result) > 0 else None #Devuelvo el ultimo valor de ese array. Si no hay ninguno, devuelvo None.

def display_heatmap(matrix):
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()

def extract_perimeter(mask_image):
    mask_image=mask_image.astype(np.uint8)
    # Find contours in the binary image
    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area (assuming it is the white island)
    contour = max(contours, key=cv2.contourArea)

    # Create a black image as the canvas
    canvas = np.zeros_like(mask_image)

    # Draw the perimeter of the contour on the canvas
    perimeter_image = cv2.drawContours(canvas, [contour], 0, (255, 255, 255), 1)

    return ((perimeter_image/255).astype(np.uint8))

def calculacentro(mask):
    M = cv2.moments(mask)
    cY = int(M["m10"] / M["m00"])
    cX = int(M["m01"] / M["m00"])
    centro=np.zeros(2)
    centro[0]=cX
    centro[1]=cY
    return(centro.astype(np.int32))

def perimetrocoord(mask):
    erodedmask=cv2.erode(mask, np.ones((3, 3), np.uint8)) #Suavizo la máscara.
    interiormask=cv2.erode(erodedmask, np.ones((3, 3), np.uint8)) #Me como el bode de la mascara suavizada
    contorno=erodedmask-interiormask #Al restar las mascaras, me quedaré con el borde.
    white_pixel_coordinates = np.argwhere(contorno == 1)
    
    # Center in origin. Find the center and compute the difference.
    M = cv2.moments(mask)
    cY = int(M["m10"] / M["m00"])
    cX = int(M["m01"] / M["m00"])
    centro=np.zeros(2)
    centro[0]=cX
    centro[1]=cY
    
    # Extract the X and Y coordinates.
    x_coordinates = white_pixel_coordinates[:, 0]-cX
    y_coordinates = white_pixel_coordinates[:, 1]-cY
    perimeter_points=np.zeros((len(x_coordinates),2))
    perimeter_points[:,0]=x_coordinates.copy()
    perimeter_points[:,1]=y_coordinates.copy()
    
    #Conversion a polares.
    perimeter_polar=np.zeros_like(perimeter_points).astype(np.float32) #creo el array
    perimeter_polar[:, 0]=np.sqrt((perimeter_points[:, 0]*perimeter_points[:, 0])+(perimeter_points[:, 1]*perimeter_points[:, 1])) #Coordenada R
    perimeter_polar[:, 1]=np.arctan2(perimeter_points[:, 1],perimeter_points[:, 0]) #Calcula theta en radianes

    perimeter_polar=perimeter_polar[perimeter_polar[:, 1].argsort()]
    
    #Reconversión a cartesianas, ordenados por theta.
    perimeter_cartesianas=np.zeros_like(perimeter_points).astype(np.float32) #creo el array
    perimeter_cartesianas[:, 0]=perimeter_polar[:, 0]*np.cos(perimeter_polar[:, 1])+cX
    perimeter_cartesianas[:, 1]=perimeter_polar[:, 0]*np.sin(perimeter_polar[:, 1])+cY
    
    # Extract the polar coordinates
    return(perimeter_polar, perimeter_cartesianas.astype(np.int32),centro)

def construye_recta(centro,puntocontorno,puntocontornopolares): #Hay que introducir el centro en cartesianas, y el punto destino (el que está en el contorno) tanto en carteianas como en polares. Necesitamos la coordenada 'R'.
    error=True
    centro=centro.astype(np.float32)
    puntocontorno=puntocontorno.astype(np.float32)
    puntocontornopolares=puntocontornopolares.astype(np.float32)
    if ((centro[0]>puntocontorno[0]) and (centro[1]!=puntocontorno[1])): #Si la coordenada X del centro es superior que la del destino...
        puntos_x=np.linspace(puntocontorno[0],centro[0],puntocontornopolares[0].astype(np.uint32)) #Las coordenadas X de la recta que usaremos para calcular el perfil, se encontrarán entre las coordenadas X del centro y del destino, y habrá tantas como distancia haya del centro al punto destino (Coordenada R). Eso sí, habrá que redondear, y astype lo hace siempre a la baja.
        puntos_y=(centro[0] * puntocontorno[1] - centro[1] * puntocontorno[0] + puntos_x * (centro[1] - puntocontorno[1])) / (-puntocontorno[0] + centro[0]) #Esto es la ecuacion de la recta que pasa por dos puntos, despejada para 'y'. Dado un array con las coordenadas 'x' y los dos puntos por los que pasa, saca un array con las coordenadas 'y' correspondientes.
        error=False
    if ((centro[0]<puntocontorno[0]) and (centro[1]!=puntocontorno[1])): #Si ocurre lo contrario, tendré que escribir el array al reves, o devolverá un array vacío en X.
        puntos_x=np.linspace(centro[0],puntocontorno[0],puntocontornopolares[0].astype(np.uint32))
        puntos_y=(centro[0] * puntocontorno[1] - centro[1] * puntocontorno[0] + puntos_x * (centro[1] - puntocontorno[1])) / (-puntocontorno[0] + centro[0])
        error=False
    else: #Si se ejecuta esto, es que estoy a 0, 180, 90 o -90 grados, y las coordenadas x o y de los dos puntos coinciden. Si se ejecutase arriba, daría error.
        if (centro[0]==puntocontorno[0]): #Si las coordenadas X coinciden, estoy a 90 o -90 grados
            puntos_x=np.ones(puntocontornopolares[0].astype(np.uint32))*centro[0] #Preparo un array de la longitud adecuada lleno del mismo valor para todas las coordenadas X
            if (centro[1]<puntocontorno[1]): #Si la coordenada Y de destino es mayor que la de origen, estoy en theta=90 grados
                puntos_y=np.arange(centro[1].astype(np.uint32),puntocontorno[1].astype(np.uint32)) #Las coordenadas I serán enteros (pixeles), que van aumentando el centro hasta el destino
                error=False
            if (centro[1]>puntocontorno[1]): #Si la coordenada Y de destino es mayor que la de origen, estoy en theta=-90 grados
                puntos_y=np.arange(puntocontorno[1].astype(np.uint32),centro[1].astype(np.uint32))
                error=False
        if (centro[1]==puntocontorno[1]): #Si las coordenadas Y coinciden, estoy a 0 o 180 grados. Simplemente cambio los papeles de la X y de la Y, y listo.
            puntos_y=np.ones(puntocontornopolares[0].astype(np.uint32))*centro[1]
            if (centro[0]<puntocontorno[0]): #Si ocurre esto, estoy a theta=0 grados
                puntos_x=np.arange(centro[0].astype(np.uint32),puntocontorno[0].astype(np.uint32))
                error=False
            if (centro[0]>puntocontorno[0]): #Si ocurre esto, estoy a 180 grados
                puntos_x=np.arange(puntocontorno[0].astype(np.uint32),centro[0].astype(np.uint32))
                error=False
    if (error==False):
        resultado=np.zeros((puntocontornopolares[0].astype(np.uint32),2)).astype(np.int32)
        resultado[:,0]=puntos_x.astype(np.int32)
        resultado[:,1]=puntos_y.astype(np.int32) #Meto los punto en formato int32, para eliminar posibles decimales del tipo 169.00
        return(resultado) #Devuelvo una matriz, donde cada fila, de dos elementos, son las coordenadas X,Y de cada punto de la recta que une el centro y el perímetro.
    else: #Si error no es FALSE, ha habido un error, por ejemplo, que el punto inicial y final sea el mismo.
        print("ERROR EN EL CÁLCULO DE LOS PERFILES")
        resultado=np.zeros((puntocontornopolares[0].astype(np.uint32),2)).astype(np.int32)
        return(resultado) #Aviso que algo ha ido mal y devuelvo todo ceros.

def construye_recta2(centro, puntocontorno, puntocontornopolares):
    centro = centro.astype(np.float32)
    puntocontorno = puntocontorno.astype(np.float32)
    puntocontornopolares = puntocontornopolares.astype(np.float32)
    
    error = True
    
    if centro[0] != puntocontorno[0] or centro[1] != puntocontorno[1]:
        if centro[0] > puntocontorno[0]:
            puntos_x = np.linspace(puntocontorno[0], centro[0], num=np.round(puntocontornopolares[0]).astype(np.uint32))
        else:
            puntos_x = np.linspace(centro[0], puntocontorno[0], num=np.round(puntocontornopolares[0]).astype(np.uint32))
        
        puntos_y = (centro[0] * puntocontorno[1] - centro[1] * puntocontorno[0] + puntos_x * (centro[1] - puntocontorno[1])) / (-puntocontorno[0] + centro[0])
        
        error = False
    else:
        if centro[1] < puntocontorno[1]:
            puntos_y = np.arange(centro[1].astype(np.uint32), puntocontorno[1].astype(np.uint32))
        elif centro[1] > puntocontorno[1]:
            puntos_y = np.arange(puntocontorno[1].astype(np.uint32), centro[1].astype(np.uint32))
        elif centro[0] < puntocontorno[0]:
            puntos_x = np.arange(centro[0].astype(np.uint32), puntocontorno[0].astype(np.uint32))
        elif centro[0] > puntocontorno[0]:
            puntos_x = np.arange(puntocontorno[0].astype(np.uint32), centro[0].astype(np.uint32))
        
        error = False
    
    if not error:
        resultado = np.zeros((len(puntos_x), 2), dtype=np.int32)
        resultado[:, 0] = puntos_x.astype(np.int32)
        resultado[:, 1] = puntos_y.astype(np.int32)
        return resultado
    else:
        print("ERROR EN EL CÁLCULO DE LOS PERFILES")
        resultado = np.zeros((puntocontornopolares[0].astype(np.uint32), 2), dtype=np.int32)
        return resultado

def generate_gaussian_matrix(matrix_size, center):
    # Create a meshgrid of x and y coordinates
    x = np.arange(matrix_size[1])
    y = np.arange(matrix_size[0])
    xx, yy = np.meshgrid(x, y)

    # Calculate the distance from each point to the center
    distance = np.sqrt((xx - center[1])**2 + (yy - center[0])**2)

    # Calculate the Gaussian values based on the distance
    sigma = matrix_size[0] / 6  # Adjust the sigma value as needed
    gaussian = np.exp(-0.5 * (distance / sigma)**2)

    # Normalize the Gaussian matrix
    gaussian /= np.max(gaussian)

    # Set the center value to 1
    gaussian[center[0], center[1]] = 1.0

    return gaussian

def FAZ_Definitiva(FAZ_Preparada, mascara, threshold): #Aquí hay que meter la faz ya recortada, suavizada con convolución, y acotada entre 0 y 255. También hay que meter su máscara binaria. Esto lo incluire dentro de otro subalgoritmo que se encargue de hacer todo bien. Threshold es el porcentaje de la derivada máxima que es necesario para que se marque el pixel en verde.
    perimetro_polar,perimetro_cartesianas,centro=perimetrocoord(mascara)
    perfiles=list() #Creo una lista de arrays, donde cada array es un perfil. No puedo definir un array porque unos perfiles erán mas largos que otros, al menos que sea un circulo perfecto, claro.
    coordenadaspintadas=list()
    mascarafinal=np.zeros_like(FAZ_Preparada) #Preparo una imagen donde pintaré la máscara
    for i in range(perimetro_cartesianas.shape[0]): #Para cada punto del contorno...
        coordenadas=construye_recta(centro,perimetro_cartesianas[i,:],perimetro_polar[i,:]) #Calcula la recta que une el centro y el punto 'i' del contorno.
        perfil_intensidades=FAZ_Preparada[coordenadas[:,0],coordenadas[:,1]] #Esto extrae el perfil de los píxeles que acabamos de calcular, que son los que conforman la recta desde el centro hasta un punto del perimetro.
        puntoapintar=give_position(np.diff(perfil_intensidades),threshold) #Calculo la derivada del perfil y busco el último pixel en el que la derivada es un threshold % del máximo.
        mascarafinal[coordenadas[0:puntoapintar,0],coordenadas[0:puntoapintar,1]]=255 #Cojo desde del centro hasta ese pixel objetivo, y pinto de blanco en la máscara ese trozo de la recta. En cada iteración, se pinta una recta más, así hasta acabar.
        coorditeracion=list()
        coorditeracion.append(coordenadas[0:puntoapintar,0]) #Coordenadas X que he pintado en esta iteración
        coorditeracion.append(coordenadas[0:puntoapintar,1]) #Coordenadas Y que he pintado en esta iteración
        coordenadaspintadas.append(coorditeracion) #Me guardo las coordenadas que he pintado
        resultado_iteracion=list() #Voy también a organizar y devolver los perfiles. Devolveré una lista de lostas, donde cada sublista es el resultado de una iteración.
        resultado_iteracion.append(perfil_intensidades) #Primero, meto el perfil de intensidades que ha salido
        resultado_iteracion.append(perimetro_polar[i,1]) #Luego, meto, junto al perfil, el ángulo theta para el que se ha obtenido. Así, están clasificados.
        perfiles.append(resultado_iteracion) #Y por último, meto la sublista en la lista donde guardaré todos los perfiles. Al final, será una lista de tantos elementos como píxeles hay en el contorno, y, a su vez, cada elemento es una lista con dos elementos: El perfil y el ángulo theta que le corresponde, en radianes.
    return(mascarafinal,perfiles,coordenadaspintadas) #Devuelto tanto los perfiles, como la máscara final, una vez se han pintado todos los píxeles que corresponden

def FAZ_Definitiva2(FAZ_Preparada, mascara, threshold):
    perimetro_polar, perimetro_cartesianas, centro = perimetrocoord(mascara)
    perfiles = []
    coordenadaspintadas = []
    mascarafinal = np.zeros_like(FAZ_Preparada)
    
    for i in range(perimetro_cartesianas.shape[0]):
        coordenadas = construye_recta2(centro, perimetro_cartesianas[i, :], perimetro_polar[i, :])
        
        # Verificar si las coordenadas están dentro de los límites de la imagen
        mask = (coordenadas[:, 0] >= 0) & (coordenadas[:, 0] < FAZ_Preparada.shape[0]) & \
               (coordenadas[:, 1] >= 0) & (coordenadas[:, 1] < FAZ_Preparada.shape[1])
        
        if np.any(mask):
            coordenadas_validas = coordenadas[mask, :]
            
            perfil_intensidades = FAZ_Preparada[coordenadas_validas[:, 0], coordenadas_validas[:, 1]]
            puntoapintar = give_position(np.diff(perfil_intensidades), threshold)
            
            mascarafinal[coordenadas_validas[0:puntoapintar, 0], coordenadas_validas[0:puntoapintar, 1]] = 255
            
            coorditeracion = [coordenadas_validas[0:puntoapintar, 0], coordenadas_validas[0:puntoapintar, 1]]
            coordenadaspintadas.append(coorditeracion)
            
            resultado_iteracion = [perfil_intensidades, perimetro_polar[i, 1]]
            perfiles.append(resultado_iteracion)
    
    return mascarafinal, perfiles, coordenadaspintadas

def parabola(x,a,binlimite,valorlimite):
    return ((2 * a - 1) * binlimite ** 2 - 2 * binlimite * a * np.array(x) + a * np.array(x) ** 2 + valorlimite)

def sigmoide(x,a,c,M,m):
    return np.exp(a * (x - c)) * (M - m) / (np.exp(a * (x - c)) + 1) + m

def ajustaperfil2(perfil):
    resultado = [0, 0, 0, 0, 0]
    
    # Primero, probamos con la sigmoide.
    fit_func = lambda x, a, c: sigmoide(x, a, c, perfil.item(len(perfil) - 1), perfil.item(0))
    
    try:
        # Ajustamos la curva sigmoide
        popt, pcov = curve_fit(fit_func, range(len(perfil)), perfil, maxfev=600)
        
        # Separar los valores de 'a' y 'c'
        a_fit, c_fit = popt
        
        # Calcular el valor de R^2
        y_pred = fit_func(range(len(perfil)), a_fit, c_fit)
        ssr = np.sum((perfil - y_pred) ** 2)
        sst = np.sum((perfil - np.mean(perfil)) ** 2)
        r_sigmoide = 1 - (ssr / sst)
        
        # Los primeros 3 valores son a, c y R^2 del ajuste a sigmoide.
        resultado[0:3] = [a_fit, c_fit, r_sigmoide]
    except RuntimeError:
        # No se pudo ajustar la curva sigmoide, saltar a la siguiente iteración
        return None
    
    # Probamos con la parábola
    fit_func = lambda x, a: parabola(x, a, perfil.argmin(), perfil.min())
    
    try:
        # Ajustamos la curva parábola
        popt, pcov = curve_fit(fit_func, range(len(perfil)), perfil, maxfev=600)
        
        # Valor de a de la parábola
        a_fit = popt[0]
        
        # Calcular el valor de R^2
        y_pred = fit_func(range(len(perfil)), a_fit)
        ssr = np.sum((perfil - y_pred) ** 2)
        sst = np.sum((perfil - np.mean(perfil)) ** 2)
        r_parabola = 1 - (ssr / sst)
        
        # El cuarto valor es a y el quinto es R^2 del ajuste a parábola.
        resultado[3:5] = [a_fit, r_parabola]
    except RuntimeError:
        # No se pudo ajustar la curva parábola, saltar a la siguiente iteración
        return None
    
    if resultado[2] > resultado[4]:  # Si R^2 de la sigmoide es mayor
        if 0.3 < resultado[0] < 10:  # Valor de 'a' entre 0.3 y 10 (evitar líneas rectas)
            if 0 <= resultado[1] < len(perfil):  # Valor central de la sigmoide entre 0 y la máxima x
                return (resultado[0] * resultado[1] + math.log(max(-(perfil[len(perfil) - 1] * 0.75 - perfil[0]) / perfil[len(perfil) - 1] / (0.75 - 1), 0.0001))) / resultado[0]  # Devolver valor calculado
    
    if resultado[4] > resultado[2]:  # La parábola ajusta mejor
        if 0 <= fit_func(0, resultado[3]) < 400 and 0 <= fit_func(len(perfil) - 1, resultado[3]) < 400:  # Extremos de la parábola en rango
            if -100 <= fit_func(0, resultado[3]) <= 0 and -100 <= fit_func(len(perfil) - 1, resultado[3]) <= 0:  # Extremos en rango negativo
                return 1.2 * perfil.argmin()  # Devolver valor apartado del mínimo
    
    # No se pudo determinar el número de perímetros que hay que dejar
    return None


def separaperimetros(FazPreparada,mascara): #Aquí lo que hago es calcular el promedio de intensidades de cada perimetro, y devuelvo un array con los resultados. La mascara ha de ser de ceros y unos. FazPreparada es la FAZ recortada de la imagen y promediada.
    mascara=mascara/mascara.max()
    mascara=mascara.astype(np.float64) #Convierto todos los valores a float, porque si no, hay errores al redondear.
    mascararestante=mascara.copy() #Aquí, ire restando perimetro tras perímetro. Basicamente es una erosion, pero controlo que pixeles quito para poder analizar más la imagen.
    resultado=list()
    
    while (mascararestante.sum()>0): #Mientras me quede algo de área
        mascararestar=extract_perimeter(mascararestante) #Extraigo el perímetro de la máscara restante
        puntosapintar=np.where(mascararestar==1) #Me quedo con las coordenadas de los píxeles del perímetro
        resultado.append(np.average(FazPreparada[puntosapintar[0],puntosapintar[1]])) #Cojo los niveles de intensidad de esos puntos que caen en el perímetro, y calculo el promedio. Meto ese valor en el array 'resultado'
        mascararestante=mascararestante-mascararestar #Elimino el perímetro de la máscara resultante, que quedará erosionada en una unidad. Al repetir, el siguiente perímetro será mas interior, y así sigo hasta que me quede sin área.
    
    resultado=np.array(resultado) #Convierto la lista a array
    mascara=mascara.astype(np.uint8) #Devuelvo al formato original
    return(np.flip(resultado)) #Doy la vuelta al array para que los primeros numeros sean las intensidades internas, y los del fiunal, las mas externas.
        
def pintadebug(mascara): #Aquí lo que hago es calcular el promedio de intensidades de cada perimetro, y devuelvo un array con los resultados. La mascara ha de ser de ceros y unos. FazPreparada es la FAZ recortada de la imagen y promediada.
    i=0
    mascararestante=mascara.copy() #Aquí, ire restando perimetro tras perímetro. Basicamente es una erosion, pero controlo que pixeles quito para poder analizar más la imagen.
    resultado=np.zeros_like(mascara)
    
    while (mascararestante.sum()>0): #Mientras me quede algo de área
        mascararestar=extract_perimeter(mascararestante) #Extraigo el perímetro de la máscara restante
        puntosapintar=np.where(mascararestar==1) #Me quedo con las coordenadas de los píxeles del perímetro
        resultado[puntosapintar[0],puntosapintar[1]]=5*i
        mascararestante=mascararestante-mascararestar #Elimino el perímetro de la máscara resultante, que quedará erosionada en una unidad. Al repetir, el siguiente perímetro será mas interior, y así sigo hasta que me quede sin área.
        if(i==51):
            i=0
        else:
            i=i+1 #diente de sierra
        #print(mascararestante.sum()) #debug
    return(resultado) #Doy la vuelta al array para que los primeros numeros sean las intensidades internas, y los del fiunal, las mas externas.

def segmenta(img): #Aqui ya solo hay que meter la imagen original. Lo hace todo él.
    img2=detectabordes(img,17,19) #Detecto bordes
    resultado=detectaFAZ(img2,0.25,0.25)/255 #Calculo la máscara y hago que sean solo valores 0 o 1, para que al multiplicar por la imagen original, aisle la faz
    FAZSola=img*resultado #Multiplico la mascara en la imagen para aislar la faz
    posicioncentro=calculacentro(resultado)
    factorgaussiano=np.zeros_like(img).astype(np.float32)
    factorgaussiano[posicioncentro[0]-10:posicioncentro[0]+10,posicioncentro[1]-10:posicioncentro[1]+10]=generate_gaussian_matrix((20, 20), [10, 10]) #Esto crea una gaussiana centrada en el centro de la máscara de radio 10, y máximo en el centro.
    factorgaussiano=1-factorgaussiano #Ahora, ivierto todo, para que al multiplicar, todo se quede igual (1), excepto la zona luminosa del centro, que es la que tiene valores cercanos a 0.
    
    
    KernSize=3 #Tamaño del nucleo para suavizar. Ha de ser impar.
    ConvKernel=np.ones((KernSize,KernSize))/(KernSize*KernSize) #Si divido entre N^2, los valores ya seguirán siendo entre 0 y 255.
    promediosFAZ=cv2.filter2D(FAZSola,-1,ConvKernel)
    
    promediosFAZ=promediosFAZ*resultado #Vuelvo a recortar con la máscara. Si se quiere aplicar el factor gaussiano, añadirlo multiplicando aquí.
    return(FAZ_Definitiva(promediosFAZ, resultado, 0.95)) #Como devuelo lo que sale de la función 'FAZ_Definitiva', los datos de salida son los mismos: La lista de listas y la máscara definitiva.
    
def get_tiff_files(directory):
    tiff_files = []
    for file in os.listdir(directory):
        if file.endswith(".tif"):
            tiff_files.append(file)
    return tiff_files

def histogramasimple(arrayinicial):
    resultado=np.zeros((255,len(arrayinicial)))
    for i in range(len(arrayinicial)):
        resultado[(255-arrayinicial[i]):255,i]=255
    return(resultado)

"""Y ahora, una vez están definidas todas las funciones, las ejecutamos y sacamos el resultado. Si presionas CTRL+F8 con esta celda seleccionada, ejecutarás todas las celdas anteriores automáticamente. Será instantáneo."""

directory_path = os.path.dirname(os.path.abspath(__file__))
tiff_files_list = get_tiff_files(directory_path) #Esto devuelve una lista con todos los archovis .tif

archivosvalidos=list()
for i in range(len(tiff_files_list)):
    if (tiff_files_list[i][-12]=='1'):
        archivosvalidos.append(tiff_files_list[i]) #En esta nueva lista agrupo solo los plexos superficiales.
del tiff_files_list #borro la lista inicial. No la necesito más.

i=0
resultados=list()
for name in archivosvalidos:
    img=cv2.imread(name,0)
    img2=detectabordes(img,17,19) #Detecto bordes
    resultado=detectaFAZ(img2,0.25,0.25)/255 #Calculo la máscara y hago que sean solo valores 0 o 1, para que al multiplicar por la imagen original, aisle la faz
    FAZSola=img*resultado #Multiplico la mascara en la imagen para aislar la faz
    
    KernSize=3 #Tamaño del nucleo para suavizar. Ha de ser impar.
    ConvKernel=np.ones((KernSize,KernSize))/(KernSize*KernSize) #Si divido entre N^2, los valores ya seguirán siendo entre 0 y 255.
    promediosFAZ=cv2.filter2D(FAZSola,-1,ConvKernel)
    
    promediosFAZ=promediosFAZ*resultado
    PerfilPromedio=separaperimetros(promediosFAZ,resultado) #Calculo el perfil de la FAZ
    histograma=histogramasimple(PerfilPromedio.astype(np.uint8))
    #histograma=cv2.resize(histograma, (1000, 1000)) #Hago esto para que todas tengan el mismo tamaño
    #cv2.imwrite(str(i)+".jpg",histograma) #Guardo una imagen de histograma simple
    resultados.append(ajustaperfil2(PerfilPromedio))
    print(i)
    i=i+1