# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:57:04 2020

@author: carlausss
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from provaFunctions import check_correct_coordinates, k_means_cluster, gmm_cluster
from provaFunctions import fill_matricies_with_original_data, fill_matricies_with_smoother_data, fill_matricies_with_kMeansCluster_data, fill_matricies_with_gmmCluster_data
from provaFunctions import fill_matricies_with_kMeansCluster_AveragedData, fill_matricies_with_gmmCluster_AveragedData
    
#Example of a correct DataFrame
data = pd.DataFrame(np.random.randint(1,100,size=(100, 5)), columns=('X', 'Y','Alfa','Beta', 'Gamma'))

    #Example of an incorrect DataFrame
#data = pd.DataFrame(np.random.randint(-100,0,size=(100, 5)), columns=('X', 'Y','Alfa','Beta', 'Gamma'))

    #Example of upload by excel file
#data = pd.read_excel("c:/Users/Desktop/any_file_path/Example.xlsx")

check_correct_coordinates(data)

km = k_means_cluster(data, 2, 'Alfa')   
gmm = gmm_cluster(data, 2, 'Alfa')

od = fill_matricies_with_original_data(data, 'Alfa')
sd = fill_matricies_with_smoother_data(data, 'Alfa') 

kmd = fill_matricies_with_kMeansCluster_data(data, km) 
gmmd = fill_matricies_with_gmmCluster_data(data, gmm)

km_av = fill_matricies_with_kMeansCluster_AveragedData(data, km)
gmm_av = fill_matricies_with_gmmCluster_AveragedData(data, gmm)

def print_images(m):           

    #Generates subplots for matricies that show the original values for every parameters
    
    fig = plt.figure(figsize=(15, 25))
    ax = []
    
    ax.append( fig.add_subplot(1, 1, 1) )
    
    ax[-1].set_xlabel('X')
    ax[-1].set_ylabel('Y')    
    
    plt.imshow(m[0])
        
    ax[0].set_title(" Values")
    

    
print_images(od) 

print_images(sd) 
    
print_images(kmd) 
    
print_images(gmmd)

print_images(km_av)

print_images(gmm_av) 
    
    
    