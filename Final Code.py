# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 10:10:11 2023

@author: villa
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import os
import fnmatch
import os.path
from scipy.optimize import curve_fit

def clip(result):
    result2=result-result.min()
    result2=result2/result2.max()
    return(255*result2)
    

def gaussian_dist(length,avg,std):
    base=range(length)
    function=np.zeros_like(base).astype(np.float32)
    for i in range(length):
        function[i]=np.exp(-((base[i]-avg)*(base[i]-avg))/(2*std*std))
    return(function)

def CrossMask(resolution,std): #A mask with a '+' shape
    mask=np.zeros((resolution,resolution))
    for i in range(resolution):
        mask[:,i]=mask[:,i]+gaussian_dist(resolution,resolution/2,std)
        mask[i,:]=mask[i,:]+gaussian_dist(resolution,resolution/2,std)
    return(mask)

def Xmask(resolution,std): #A mask with an 'x' shape
    mask=np.zeros((resolution,resolution))
    for i in range(resolution):
        mask[:,i]=mask[:,i]+gaussian_dist(resolution,i,std)
        mask[i,:]=mask[i,:]+gaussian_dist(resolution,i,std)
        mask[:,i]=mask[:,i]+gaussian_dist(resolution,resolution-i,std)
        mask[i,:]=mask[i,:]+gaussian_dist(resolution,resolution-i,std)
    return(mask/2)

def LeftRightMask(resolution,std): #A mask with an '\' shape
    mask=np.zeros((resolution,resolution))
    for i in range(resolution):
        mask[:,i]=mask[:,i]+gaussian_dist(resolution,i,std)
    return(mask)

def RightLeftMask(resolution,std): #A mask with an '/' shape
    mask=np.zeros((resolution,resolution))
    for i in range(resolution):
        mask[:,i]=mask[:,i]+gaussian_dist(resolution,resolution-i,std)
    return(mask)

def Xmask_repaired(resolution, std): #Another mask with an 'x' shape, with fixed values in the center, so they are not greater than 1.
    Xmask=np.zeros([resolution,resolution,3]) #An NxNx3 matrix is initialized. In the two first 'z' coordinates, the masks defined above are initialized.
    Xmask[:,:,0]=LeftRightMask(resolution,std)
    Xmask[:,:,1]=RightLeftMask(resolution,std)
    Xmask[Xmask<(200/255)]=0 # As the gaussian profile is infinite, we floor low values so the tails do not becone noise in the final mask.
    for i in range(resolution):
        for j in range(resolution):
            if (Xmask[i,j,0]!=0 and Xmask[i,j,1]!=0):
                Xmask[i,j,2]=(Xmask[i,j,0]+Xmask[i,j,1])/4 #Here, we separately analyze the two masks separately, and, for those pixels where there is a superposition, we compute an amount equal to one fourth of the total value in the intersection. When substracting it twice, to both the masks: (Xmask[:,:,0]-Xmask[:,:,2])+(Xmask[:,:,1]-Xmask[:,:,2])=Xmask[:,:,0]+Xmask[:,:,1]-2*Xmask[:,:,2], which will substract two fourths of the total intensity in the center, which is what we want to do, so we avoid superposition issues.
                
    Xmask[:,:,0]=Xmask[:,:,0]-Xmask[:,:,2]
    Xmask[:,:,1]=Xmask[:,:,1]-Xmask[:,:,2]
    return(Xmask[:,:,0]+Xmask[:,:,1]) #We do as explained above

def CrossMask_repaired(resolution, std): #Same procedure, for the Cross-shaped mask.
    mask=np.zeros([resolution,resolution,3]) 
    for i in range(resolution):
        mask[:,i,0]=mask[:,i,0]+gaussian_dist(resolution,resolution/2,std) 
        mask[i,:,1]=mask[i,:,1]+gaussian_dist(resolution,resolution/2,std) 
    mask[mask<(200/255)]=0 
    for i in range(resolution):
        for j in range(resolution):
            if (mask[i,j,0]!=0 and mask[i,j,1]!=0):
                mask[i,j,2]=(mask[i,j,0]+mask[i,j,1])/4
                
    mask[:,:,0]=mask[:,:,0]-mask[:,:,2]
    mask[:,:,1]=mask[:,:,1]-mask[:,:,2]
    return(mask[:,:,0]+mask[:,:,1])

def Final_Mask(resolution,std): #The final mask is obtained by doing the same procedure once again, this time, with the two corrected masks we have obtained before.
    mask=np.zeros([resolution,resolution,3])
    mask[:,:,0]=CrossMask_repaired(resolution, std)
    #mask[:,:,1]=Xmask_repaired(resolution, std) #Original version.
    mask[:,:,1]=Xmask_repaired(resolution, std*1.4142) #Corrected factor. It can yield more consistent thicknesses for the lines in the 'x' mask. The performance does not vary in any perceptible quantity.
    mask[mask<(200/255)]=0
    for i in range(resolution):
        for j in range(resolution):
            if (mask[i,j,0]!=0 and mask[i,j,1]!=0):
                mask[i,j,2]=(mask[i,j,0]+mask[i,j,1])/4
                
    mask[:,:,0]=mask[:,:,0]-mask[:,:,2]
    mask[:,:,1]=mask[:,:,1]-mask[:,:,2]
    return(mask[:,:,0]+mask[:,:,1])

def Faz_Convolution(img,resolution,std,threshold):
    #Kernel X
    conv4=cv2.filter2D(img/255, -1, Final_Mask(resolution,std)) #Probability map of bifurcation calculation.
    ret, thresholded=cv2.threshold(conv4,threshold*conv4.max(),255,cv2.THRESH_BINARY_INV) #The probability map is binarized to obtain the binary mask
    return(thresholded)

def search_files(directory):
    extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'tif', 'webp']
    extensions = [f".{ext.lower()}" for ext in extensions]
    create_output_directory_tree(directory, "output")
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(fnmatch.fnmatch(file.lower(), f"*{ext}") for ext in extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths

def create_output_directory_tree(input_folder, output_folder_name="output"):
    input_folder = os.path.abspath(input_folder)
    output_base = os.path.join(os.path.dirname(input_folder), output_folder_name)
    os.makedirs(output_base, exist_ok=True)
    for root, dirs, _ in os.walk(input_folder):
        relative_path = os.path.relpath(root, input_folder)
        output_dir = os.path.join(output_base, relative_path)
        os.makedirs(output_dir, exist_ok=True)

def superposedoubleFAZ(img,FinalResult,FinalResult2,transparency): #Draws two binarized masks on top of the original image.
  A=img.copy()
  FinalResultcolor=np.zeros((A.shape[0],A.shape[1],3))
  FinalResultcolor[:,:,2]=FinalResult 
  FinalResultcolor[:,:,1]=FinalResult2 #Draws the information of the masks in the color channels.
  for i in range(FinalResultcolor.shape[0]):
    for j in range(FinalResultcolor.shape[1]):
      for k in range(FinalResultcolor.shape[2]):
        if (FinalResultcolor[i,j,k]<128): #Where there is no mask, we draw the original image.
          FinalResultcolor[i,j,k]=A[i,j]
        else:
          FinalResultcolor[i,j,k]=transparency*A[i,j]+(1-transparency)*FinalResultcolor[i,j,k] #Where there are masks, we blend the image with the masks, inserting them as color channels with a certain transparency set by user. Default: 35%.
  return(FinalResultcolor)

def Iterative_Segmentation(names): #Given a list of images and manual segmentations, creates an array of arrays, where the information of the FAZ of each image is saved. This is the main algorithm.
    OuterAreas=list()
    InternalAreas=list()
    InternalAcircularities=list()
    OuterAcircularities=list()
    OuterPerimeters=list()
    InnerPerimeters=list()
    Inner_Ratios=list()
    Outer_Ratios=list()
    for i in range(len(names)):
        try:
            #print(names[i]) #debug. Tells which image is being processed.
            img=cv2.imread(names[i],0)
            MaskConvolution=Faz_Convolution(img,45,8,0.2)
            ExternalMaskConvolution=Faz_Convolution(img,45,8,0.25)
            # Here it is where we calculate everything.
            InnerPerimeter,InnerArea,InnerAcircularity,InnerRatio,_,_=statsFAZ(MaskConvolution)
            OuterPerimeter,OuterArea,OuterAcircularity,OuterRatio,_,_=statsFAZ(ExternalMaskConvolution)
            OuterAreas.append(ExternalMaskConvolution.sum()*9/(102400*255))
            InternalAreas.append(MaskConvolution.sum()*9/(102400*255))
            InternalAcircularities.append(InnerAcircularity)
            OuterAcircularities.append(OuterAcircularity)
            OuterPerimeters.append(OuterPerimeter*3/320)
            InnerPerimeters.append(InnerPerimeter*3/320)
            Inner_Ratios.append(InnerRatio)
            Outer_Ratios.append(OuterRatio)
            #Draw Result
            Result=superposedoubleFAZ(img, MaskConvolution, ExternalMaskConvolution, 0.35)
            output_path = names[i].replace("input", "output")
            cv2.imwrite(output_path+"_Segmented.png",Result) #We save the segmented image as [OriginalName]_segmented.png
            print("Analyzed: "+str(i+1)+" Images from "+str(len(names)))
        except ValueError as e:
            print(f"Skipping image number {i + 1} due to: {e}") #Here, we pop put an error, if, by some reason, the stats could not have been computed (i.e: Defectuous image, no apparent FAZ, etc...). This prevent the program to crash.
            OuterAreas.append(0.0)
            InternalAreas.append(0.0)
            InternalAcircularities.append(0.0)
            OuterAcircularities.append(0.0)
            OuterPerimeters.append(0.0)
            InnerPerimeters.append(0.0)
            Inner_Ratios.append(0.0)
            Outer_Ratios.append(0.0)
    return(InnerPerimeters,OuterPerimeters,InternalAreas,OuterAreas,InternalAcircularities,OuterAcircularities,Inner_Ratios,Outer_Ratios) #Once we finish looping through all the images, we return all the stats, each one, an array of length total to the number of images.
        
#Here, we initialize the algorithms to compute FAZ-Related stats.

def vertical(BinaryMask): #Vertical derivative
    MaskBorder=np.zeros_like(BinaryMask)
    for j in range(BinaryMask.shape[1]):
        MaskBorder[:,j]=np.append(np.diff(BinaryMask[:,j]),0)
    MaskBorder[MaskBorder!=0]=255
    return(MaskBorder)

def horizontal(BinaryMask): #Horizontal derivative
    MaskBorder=np.zeros_like(BinaryMask)
    for i in range(BinaryMask.shape[0]):
        MaskBorder[i,:]=np.append(np.diff(BinaryMask[i,:]),0)
    MaskBorder[MaskBorder!=0]=255
    return(MaskBorder)

def Extract_Border(image): #We obtain the border by analyzing when the derivative of the binary mask is different from zero. The result is a black image with the contour of the FAZ painted in white.
    return(vertical(image)+horizontal(image))

def distance(punto1,punto2):
  return(np.sqrt((punto2[0]-punto1[0])*(punto2[0]-punto1[0])+(punto2[1]-punto1[1])*(punto2[1]-punto1[1])))

def func(x,a,b,direction): #This is the equation of an ellipse in polar coordinates. It is used to fit the axis lengths, as well as the axis ratio from a list of points depicting a closed contour.
    return(np.power((np.sin(x-direction)/a)*(np.sin(x-direction)/a)+(np.cos(x-direction)/b)*(np.cos(x-direction)/b),-1/2))

def statsFAZ(BinaryMask): # Faz_Contour is an array of length 'N' of [x,y] points, where 'N' is the number of points that form the contour.
  if(BinaryMask.sum()==0):
      return(np.zeros(6))
  perimeter=0
  Xsum=0
  Ysum=0
  area=0
  Faz_Contour = list(np.where(Extract_Border(BinaryMask) == 255))
  Faz_Contour = list(np.transpose(Faz_Contour)) #We extract the coordinates from the image, and transpose, so the loops iterate in the correct dimension.
  for i in range(len(Faz_Contour)):
      perimeter=perimeter+distance(Faz_Contour[i-1],Faz_Contour[i]) #Perimeter is the sum of the distance of all points.
      area=area+(Faz_Contour[i-1][0]*Faz_Contour[i][1]-Faz_Contour[i][0]*Faz_Contour[i-1][1])/2 #Shoelace Method for area calculation.
      Xsum=Xsum+Faz_Contour[i][0] #Needed to compute the center of masses of the FAZ.
      Ysum=Ysum+Faz_Contour[i][1]
  FAZ_Center=np.array([(Xsum/len(Faz_Contour)),(Ysum/len(Faz_Contour))])
  
  r_Coordinate=np.zeros(len(Faz_Contour)) #Now that we have defined an origin for the polar coordinates, we can convert the contour to polars.
  theta_Cooordinate=np.zeros(len(Faz_Contour)) #Coordinates here are calculated separately, so it is easier to compute acircularity later.
  Centered_Perimeter=list(Faz_Contour) #Now, we create a copy of the contour, and we will shift the coordinates, so the center of masses lays on the origin.
  for i in range(len(Faz_Contour)):
      Centered_Perimeter[i]=Faz_Contour[i]-FAZ_Center #We shift the perimeter
      r_Coordinate[i]=distance(Centered_Perimeter[i],np.array([0,0])) #Then, we compute the 'r' coordinate of the point as the distance to the origin.
      theta_Cooordinate[i]=np.arctan(Centered_Perimeter[i][1]/Centered_Perimeter[i][0]) #And the Theta coordinate too.
      
  #Finally, once the contour is expressed in polar, we can fit it with the curve_fit algorithm, in order to retrieve the missing parameters.
  acircularity=np.std(r_Coordinate)/np.average(r_Coordinate) #La acircularity es la desviación estándar de la coordenada radial.
  popt, pcov = curve_fit(func, theta_Cooordinate, r_Coordinate) #En popt aparecerán los valores ajustados de a,b y dirección en ese orden.
  return(np.array([perimeter,area,acircularity,max((popt[0]/popt[1]),(popt[1]/popt[0])),2*popt[0],popt[2]])) #Returns the parameters in the following order: Perímeter, area, acircularity, axis ratio, major axis and direction.

def group_detection(image_path):
    return(os.path.split(os.path.split(image_path)[0])[1])

#------------------------
#Main Algorithm
#------------------------

Original_images=search_files(os.getcwd()+"/input") #Search image files in the working directory
InnerPerimeters,OuterPerimeters,InternalAreas,OuterAreas,InternalAcircularities,OuterAcircularities,Inner_Ratios,Outer_Ratios=Iterative_Segmentation(Original_images) #Run the code. Will write the images and compute all the FAZ stats, for every image in 'Original_images'.

#------------------------
#Graph Results (Settings in line 319-340)
#------------------------

#Organize measurements in groups to be used as Seaborn graph data

Inner_Ratios_MD1=list()
InnerPerimeters_MD1=list()
InternalAcircularities_MD1=list()
InternalAreas_MD1=list()
Outer_Ratios_MD1=list()
OuterAcircularities_MD1=list()
OuterAreas_MD1=list()
OuterPerimeters_MD1=list()
Inner_Ratios_MD2=list()
InnerPerimeters_MD2=list()
InternalAcircularities_MD2=list()
InternalAreas_MD2=list()
Outer_Ratios_MD2=list()
OuterAcircularities_MD2=list()
OuterAreas_MD2=list()
OuterPerimeters_MD2=list()
Inner_Ratios_Healthy=list()
InnerPerimeters_Healthy=list()
InternalAcircularities_Healthy=list()
InternalAreas_Healthy=list()
Outer_Ratios_Healthy=list()
OuterAcircularities_Healthy=list()
OuterAreas_Healthy=list()
OuterPerimeters_Healthy=list()

for i in range(len(Original_images)): #We split the previous results by groups
    if(group_detection(Original_images[i])=='MD1'):
        Inner_Ratios_MD1.append(Inner_Ratios[i])
        InnerPerimeters_MD1.append(InnerPerimeters[i])
        InternalAcircularities_MD1.append(InternalAcircularities[i])
        InternalAreas_MD1.append(InternalAreas[i])
        Outer_Ratios_MD1.append(Outer_Ratios[i])
        OuterAcircularities_MD1.append(OuterAcircularities[i])
        OuterAreas_MD1.append(OuterAreas[i])
        OuterPerimeters_MD1.append(OuterPerimeters[i])
        
    elif(group_detection(Original_images[i])=='MD2'):
        Inner_Ratios_MD2.append(Inner_Ratios[i])
        InnerPerimeters_MD2.append(InnerPerimeters[i])
        InternalAcircularities_MD2.append(InternalAcircularities[i])
        InternalAreas_MD2.append(InternalAreas[i])
        Outer_Ratios_MD2.append(Outer_Ratios[i])
        OuterAcircularities_MD2.append(OuterAcircularities[i])
        OuterAreas_MD2.append(OuterAreas[i])
        OuterPerimeters_MD2.append(OuterPerimeters[i])

    elif(group_detection(Original_images[i])=='Healthy'):
        Inner_Ratios_Healthy.append(Inner_Ratios[i])
        InnerPerimeters_Healthy.append(InnerPerimeters[i])
        InternalAcircularities_Healthy.append(InternalAcircularities[i])
        InternalAreas_Healthy.append(InternalAreas[i])
        Outer_Ratios_Healthy.append(Outer_Ratios[i])
        OuterAcircularities_Healthy.append(OuterAcircularities[i])
        OuterAreas_Healthy.append(OuterAreas[i])
        OuterPerimeters_Healthy.append(OuterPerimeters[i])
        
del i

def create_dataframe(x_coordinates, y_coordinates,TitleX,TitleY): #Handle previously generated data into Pandas dataframe, for use in Seaborn.
    data = {TitleX: x_coordinates, TitleY: y_coordinates}
    df = pd.DataFrame(data)
    return df

###Define graph characteristics

#Number of bins for the marginal histograms, graph limits, and axis titles.

MaxBins = 20

max_x =0.7 #For appropaite values of acircularity and perimeter graphs, use the limits used in the article graphs as guidance.
max_y = 1
min_x = 0
min_y = 0

TitleX='Inner Area (mm²)'
TitleY='Outer Area (mm²)'

# Define which arrays to use as graph data for each group.

x_coordinates_MD1=InternalAreas_MD1.copy()
x_coordinates_MD2=InternalAreas_MD2.copy()
x_coordinates_Healthy=InternalAreas_Healthy.copy()
y_coordinates_MD1=OuterAreas_MD1.copy()
y_coordinates_MD2=OuterAreas_MD2.copy()
y_coordinates_Healthy=OuterAreas_Healthy.copy()

# Function to create and save a plot without legend

ranges = [max_x - min_x, max_y - min_y]

def create_and_save_plot(data, color, filename, Xtitle, Ytitle, bin_width=max(ranges)/MaxBins, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y, dpi=600):
    g = sns.JointGrid(data=data, x=Xtitle, y=Ytitle, space=0, ratio=2)
    sns.scatterplot(data=data, x=Xtitle, y=Ytitle, ax=g.ax_joint, color=color, alpha=.6, edgecolor='black', linewidth=1)
    sns.regplot(data=data, x=Xtitle, y=Ytitle, ax=g.ax_joint, scatter=False, color=color)
    bin_range_x = np.arange(min_x, max_x, bin_width)
    num_bins_x = len(bin_range_x) - 1
    bin_range_y = np.arange(min_y, max_y, bin_width)
    num_bins_y = len(bin_range_y) - 1

    sns.histplot(data=data, x=Xtitle, ax=g.ax_marg_x, color=color, kde=False, bins=num_bins_x, binrange=(bin_range_x.min(), bin_range_x.max()))
    sns.histplot(data=data, y=Ytitle, ax=g.ax_marg_y, color=color, kde=False, bins=num_bins_y, binrange=(bin_range_y.min(), bin_range_y.max()))
    g.ax_joint.set(xlim=(min_x, max_x), ylim=(min_y, max_y))

    g.set_axis_labels(xlabel=Xtitle, ylabel=Ytitle, fontsize=18)

    g.ax_joint.tick_params(axis='both', which='major', labelsize=18)
    g.ax_marg_x.tick_params(axis='both', which='major', labelsize=18)
    g.ax_marg_y.tick_params(axis='both', which='major', labelsize=18)

    g.savefig(filename, dpi=dpi)
    plt.close()

# Plot creation

create_and_save_plot(create_dataframe(x_coordinates_MD1, y_coordinates_MD1,TitleX,TitleY), "blue", 'plot_dm1.png', TitleX, TitleY, dpi=600)
create_and_save_plot(create_dataframe(x_coordinates_MD2, y_coordinates_MD2,TitleX,TitleY), "red", 'plot_dm2.png',TitleX, TitleY, dpi=600)
create_and_save_plot(create_dataframe(x_coordinates_Healthy, y_coordinates_Healthy,TitleX,TitleY), "green", 'plot_healthy.png',TitleX, TitleY, dpi=600)

# Fuse three graphs into one with transparency.

img_dm1 = cv2.imread('plot_dm1.png', 1)
img_dm2 = cv2.imread('plot_dm2.png', 1)
img_healthy = cv2.imread('plot_healthy.png', 1)

stacked_image = (img_healthy * 0.2222) + (img_dm2 * 0.3333) + (img_dm1 * 0.4444)
cv2.imwrite('Final_graph.png', stacked_image)

# Remove intermediate files for more clarity.
os.remove("plot_dm1.png")
os.remove("plot_dm2.png")
os.remove("plot_healthy.png")