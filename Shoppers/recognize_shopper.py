# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 11:33:18 2017

@author: Hiram Ruiz
"""

#Packages
import _pickle as cPickle
import shutil
from wand.image import Image as Img
import os
from PIL import Image
import cv2
import pytesseract as tess
import numpy as np
from sklearn.cluster import DBSCAN
import compute_heuristics
import pandas as pd


#Define Paths:  
#Main
dir_path = 'C:\\Users\\Hiram\\Desktop\\Projects\\Shoppers\\'

#Subs
pdf_path = os.path.join(dir_path, 'Raw_Shoppers')
png_path = os.path.join(dir_path, 'Pngs')
binary_path = os.path.join(dir_path, 'Binaries')
edge_path = os.path.join(dir_path, 'Edges')

#Define Functions--------------------------------------------------------------

#Pre-Processing Images---------------------------------------------------------
def pdf_to_png(file):
   if not os.path.exists(dir_path+"Pngs"):
       os.makedirs("Pngs")

   with Img(filename=os.path.join(pdf_path,file), resolution=700) as img:
       img.compression_quality = 99
       img.save(filename='Pngs\\'+os.path.splitext(file)[0]+'.png')
                      
def to_binary(file):
    if not os.path.exists(dir_path+"Binaries"):
        os.makedirs("Binaries")
    
    
    with Image.open(os.path.join(png_path,file)) as img:
            gray = img.convert('L')
            bw = gray.point(lambda x: 0 if x<180 else 255, '1')
            bw.save('Binaries\\'+os.path.splitext(file)[0]+'.png')
                                 
def edges(file):
    if not os.path.exists(dir_path+"Edges"):
        os.makedirs("Edges")
        
   
    img = cv2.imread(os.path.join(binary_path,file),0)
    edges = cv2.Canny(img, 100, 200)
    cv2.imwrite('Edges\\'+os.path.splitext(file)[0]+'.png',edges)
     
    
#Bounding Boxes Computation----------------------------------------------------

def blob_bounds(file):
    if not os.path.exists(dir_path+"Blobs"):
        os.makedirs("Blobs")
               
    img = cv2.imread(os.path.join(edge_path,file),0)      
    image, contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
       
    boundingboxes=[]
    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)   
            
        # Area Filter
        if w > 25 and w < 500 and h > 25 and h < 600:
                     
            #Append
            boundingboxes.append([x,y,x+w,y+h])
    return(boundingboxes)

def get_centroids(boundingboxes):
    centroids=[]
    for box in boundingboxes:
        centroid=[round((box[2]+box[0])/2),round((box[3]+box[1])/2)]
        centroids.append(centroid)
    return(centroids) 
  
def group_blobs(boundingboxes, sensitivity): #DBSCAN Clustering on Centroids ---------------
    centers = get_centroids(boundingboxes)
    
    clustering=DBSCAN(eps=sensitivity, min_samples=3).fit(centers)
    labels = clustering.labels_
                    
    return (labels)

def bounding_box(coords):

  min_x = 100000 
  min_y = 100000
  max_x = -100000
  max_y = -100000

  for item in coords:
    if item[0] < min_x:
      min_x = item[0]

    if item[0] > max_x:
      max_x = item[0]

    if item[1] < min_y:
      min_y = item[1]

    if item[1] > max_y:
      max_y = item[1]

  return [(min_x,min_y),(max_x,min_y),(max_x,max_y),(min_x,max_y)]

def enclose_groups(boundingboxes, grouplist):
    clusters=[]
    unique=list(set(grouplist))
    
    for g in unique:
        blobs_to_enclose=[]
        indices = [i for i, x in enumerate(grouplist) if x == g]
        
        for ii in indices:
            blobs_to_enclose.append(boundingboxes[ii])
        
        coords=[]
        for blob in blobs_to_enclose:
            coord1=blob[:2]
            coord2=blob[2:]
            coords.append(coord1)
            coords.append(coord2)
            
            
        bbox=bounding_box(coords)
        
        clusters.append(bbox)
        
    return(clusters)

def filter_clusters(clusters, area_threshold_lower, area_threshold_upper):
    filtered_clusters=[]
    for clust in clusters:
        length=abs(clust[0][0]-clust[1][0])
        height=abs(clust[0][1]-clust[3][1])
        area=length*height
        
        if area > area_threshold_lower and area < area_threshold_upper:
            filtered_clusters.append(clust)
            
    return(filtered_clusters)
                

def draw_clusters(image, clusters):

    #Get Cluster Corners
    for clust in clusters:        
        cv2.rectangle(image, clust[0], clust[2], (200, 100, 200), 8)
            
#   Show Original Image Clusters
    cv2.imshow('filter_blobs', image)       

#------------------------------------------------------------------------------
#Break into more managable pieces:
    
def crop_clusters(img, clusters):
    if not os.path.exists(dir_path+"Pieces"):
        os.makedirs("Pieces")
        
    else:
        shutil.rmtree("Pieces")
        os.makedirs("Pieces")
        
    counter=0
    for clust in clusters:
        cropped = img[clust[0][1]:clust[3][1], clust[0][0]:clust[1][0]]
        cv2.imwrite('Pieces\\bit_'+str(counter)+'.png',cropped)
        counter=counter+1
        
#OPTICAL CHARACTER RECOGNITION-------------------------------------------------
#PRE-OCR   
def binarize(folder):
    for file in os.listdir(os.path.join(dir_path,'Pieces')):
        img=Image.open(os.path.join(dir_path,'Pieces\\'+file))
        gray = img.convert('L')
        bw = gray.point(lambda x: 0 if x<210 else 255, '1')
        bw.save('Pieces\\bw_'+os.path.splitext(file)[0]+'.png')   
        os.remove(os.path.join(dir_path,'Pieces\\'+file))
        
def vectorize_images(folder):
    compute_heuristics.comp_heur(folder,'vectorized_data')    
        
#Clear Out NON-TEXT using trained SVM calssifier
def filter_text(folder):
    
    with open('trained_classifier.pkl', 'rb') as classifier_text:
        classifier = cPickle.load(classifier_text)
        
    vectorized = pd.read_csv(dir_path+'//vectorized.csv')
    vectorized = vectorized.drop(labels='training_label', axis=1)
    predicted = classifier.predict(vectorized[:-1])   
    print(predicted)
      
    for image in os.listdir(folder):        
        if EAratio < .016:
            os.remove(os.path.join(dir_path,'Pieces\\'+image))
    
    


#OCR           
def recognize_image():
    for file in os.listdir(binary_path):
        img = Image.open(os.path.join(binary_path,file))
        recognized = tess.image_to_string(img)
        print (recognized)
       
   #Apply OCR
   #text=pytesser.image_to_string(Image.open('Jpegs\\amigo-0.jpg'))


#MAIN PROGRAM EXECUTION--------------------------------------------------------
###############################################################################

#PRE-PROCESSING
#for f in os.listdir(pdf_path):
#   file=os.path.splitext(f)[0]
#   pdf_to_png(file+'.pdf')

for f in os.listdir(png_path):
#   to_binary(f)
#   edges(f)
    
#BLOB GROUPING
    boundingboxes=blob_bounds(dir_path+'Edges\\'+f)
    grouplist=group_blobs(boundingboxes, 300)
    clusters=enclose_groups(boundingboxes, grouplist)
    clusters=filter_clusters(clusters, 200000, 10000000)
    img=cv2.imread(os.path.join(png_path,f),0)
#    image = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
#    draw_clusters(image, clusters)

#CROP BLOBS, BINARIZE and REMOVE NON-TEXT
    crop_clusters(img, clusters)
    binarize(os.path.join(dir_path,'Pieces'))
    vectorize_images(dir_path+'//Pieces')

        
            
                
    
###############################################################################   




    
