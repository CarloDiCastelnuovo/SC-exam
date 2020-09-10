# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 12:05:29 2020

@author: carlausss
"""


import numpy as np
import pandas as pd
#import pytest

from esame.Images import check_correct_coordinates, k_means_cluster, gmm_cluster, get_close_points, fill_matricies_with_original_data, fill_matricies_with_smoother_data, fill_matricies_with_kMeansCluster_data, fill_matricies_with_gmmCluster_data
from esame.Images import print_original_images, print_smoother_images, print_kMeansCluster_images, print_kMeansCluster_AveragedImages, print_gmmCluster_images, print_gmmCluster_AveragedImages
#from Images.py import check_correct_coordinates, k_means_cluster, gmm_cluster, get_close_points, fill_matricies_with_original_data, fill_matricies_with_smoother_data, fill_matricies_with_kMeansCluster_data, fill_matricies_with_gmmCluster_data

#Generate 3 different DataFrame to test the functions, df1 has correct shape df2 and df3 have not

df1 = pd.DataFrame(np.random.randint(1,100,size=(100, 5)), columns=('X', 'Y','Alfa','Beta', 'Gamma'))
   
df2 = pd.DataFrame(np.random.randint(-100,0,size=(100, 5)), columns=('X', 'Y','Alfa','Beta', 'Gamma'))
   
df3 = pd.DataFrame(np.random.uniform(size = (100,5)), columns=('X', 'Y','Alfa','Beta', 'Gamma'))
    
print("\nCorrect DataFrame shape df1: \n\n",df1.head())
    
def test_check_correct_coordinates():
    
    #Testing that the check function return True with correct DF and False with wrong DF
    
    check1 = check_correct_coordinates(df1)
    assert check1 == True
    
    check2 = check_correct_coordinates(df2)
    check3 = check_correct_coordinates(df3)
    assert check2 == False
    assert check3 == False
    

def test_get_close_points():
    
    #Testing get_close_points function gives correct number of close points for ad hoc DataFrame 
    
    dfcp = pd.DataFrame({'X' : [0,1,1,8], 'Y' : [1,0,1,5], 'test_value' : [9, 10, 11, 12]})
    
    l = []
    
    for i in range(3):    
        x = dfcp.iloc[i, 0:1]
        y = dfcp.iloc[i, 1:2]
        prova = get_close_points(dfcp, x, y, radius = 2)
        l.append(prova)
        
    assert len(l) == 3
        
    
def test_k_means_cluster():
    
    #Testing k_means_cluster generate a cluster label for every dataframe point
    
    x = k_means_cluster(df1)
    
    assert len(x) == (len(df1.iloc[0,:]) - 2)
    

def test_gmm_cluster():
    
    #Testing gmm_cluster generate a cluster label for every dataframe point
    
    x = gmm_cluster(df1)
    
    assert len(x) == (len(df1.iloc[0,:]) - 2)

def test_fill():
    
    #Testing that all the fill functions give the correct number of matricies, one for every parameter
    #for the first and second function; two for every parameter for the lasts corresponding to cluster
    #analysis on both original and averaged values.
    
    x = fill_matricies_with_original_data(df1)
    y = fill_matricies_with_smoother_data(df1)
    km = fill_matricies_with_kMeansCluster_data(df1)
    gmm = fill_matricies_with_gmmCluster_data(df1)
    
    assert len(x) == (len(df1.iloc[0,:]) - 2)
    assert len(y) == (len(df1.iloc[0,:]) - 2)
    assert len(km) == 2*(len(df1.iloc[0,:]) - 2)
    assert len(gmm) == 2*(len(df1.iloc[0,:]) - 2)
    
def test_print():
    
    #Testing that all the print functions create the correct number of subplots 
    
    assert len(print_original_images(fill_matricies_with_original_data(df1))) == (len(df1.iloc[0,:]) - 2)
    assert len(print_smoother_images(fill_matricies_with_smoother_data(df1))) == (len(df1.iloc[0,:]) - 2)
    assert len(print_kMeansCluster_images(fill_matricies_with_kMeansCluster_data(df1))) == (len(df1.iloc[0,:]) - 2)
    assert len(print_gmmCluster_images(fill_matricies_with_gmmCluster_data(df1))) == (len(df1.iloc[0,:]) - 2)
    assert len(print_kMeansCluster_AveragedImages(fill_matricies_with_kMeansCluster_data(df1))) == (len(df1.iloc[0,:]) - 2)
    assert len(print_gmmCluster_AveragedImages(fill_matricies_with_gmmCluster_data(df1))) == (len(df1.iloc[0,:]) - 2)
    


