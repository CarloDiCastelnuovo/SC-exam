# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:57:04 2020

@author: carlausss
"""

import pandas as pd
import numpy as np

from Functions import fill_matrix_with_original_data, fill_matrix_with_averaged_data, fill_matrix_with_kMeansCluster_data, fill_matrix_with_gmmCluster_data
from Functions import fill_matrix_with_kMeansCluster_AveragedData, fill_matrix_with_gmmCluster_AveragedData, check_correct_coordinates, print_image
    
    #Example of a correct DataFrame
data = pd.DataFrame(np.random.randint(1,100,size=(100, 5)), columns=('X', 'Y','Alfa','Beta', 'Gamma'))

    #Example of an incorrect DataFrame
#data = pd.DataFrame(np.random.randint(-100,0,size=(100, 5)), columns=('X', 'Y','Alfa','Beta', 'Gamma'))

    #Example of upload by excel file
#data = pd.read_excel("c:/Users/Desktop/any_file_path/Example.xlsx")

check = check_correct_coordinates(data)

if check == False:
    print("Wrong coordinates values")

print(data.head())

od_gamma = fill_matrix_with_original_data(data, 'Gamma')
sd_gamma = fill_matrix_with_averaged_data(data, 'Gamma') 
    
print_image(od_gamma, 'Gamma Values') 
print_image(sd_gamma, 'Smoothed Gamma Values') 



km_mat_alfa = fill_matrix_with_kMeansCluster_data(data, 2, 3, 2) 
gmm_mat_alfa = fill_matrix_with_gmmCluster_data(data, 2, 3, 2)

print_image(km_mat_alfa, 'K-Means Alfa Results')     
print_image(gmm_mat_alfa, 'GMM Alfa Results')



km_av_mat_beta = fill_matrix_with_kMeansCluster_AveragedData(data, 3, 4, 3) 
gmm_av_mat_beta = fill_matrix_with_gmmCluster_AveragedData(data, 3, 4, 3)

print_image(km_av_mat_beta, 'K-Means Smoothed Beta Results')
print_image(gmm_av_mat_beta, 'GMM Smoothed Beta Results') 

    
    