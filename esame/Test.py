# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:50:03 2019

@author: carlausss
"""

import numpy as np
import pandas as pd
import pytest

from esame.Functions import check_correct_coordinates, k_means_cluster, gmm_cluster, get_close_points, images, print_images

#Generate 3 different DataFrame to test the functions, df1 has correct shape df2 and df3 have not

df1 = pd.DataFrame(np.random.randint(1,100,size=(100, 5)), columns=('X', 'Y','Alfa','Beta', 'Gamma'))
   
df2 = pd.DataFrame(np.random.randint(-100,0,size=(100, 5)), columns=('X', 'Y','Alfa','Beta', 'Gamma'))
   
df3 = pd.DataFrame(np.random.uniform(size = (100,5)), columns=('X', 'Y','Alfa','Beta', 'Gamma'))
    

def test_df_shape():
    
    #Testing that check_correct_coordinates works correctly with a dataframe with correct shape
    
    check_correct_coordinates(df1)
    print("\nCorrect DataFrame shape df1: \n\n",df1.head())
    
    #Testing that check_correct_coordinates function raises error with wrong DataFrames
    
    with pytest.raises(ValueError):
        
        check_correct_coordinates(df2)
        
        check_correct_coordinates(df3)
        
test_df_shape()


def test_get_close_points():
    
    #Testing g_c_p raises ValueError for negative radius values
    
    with pytest.raises(ValueError):
        
        x = df1.iloc[:, 0]
        y = df1.iloc[:, 1]
        get_close_points(df1, x, y, radius = -2)


def test_k_means_cluster():
    
    #Testing k_means_cluster generate a cluster label for every dataframe point
    
    x = k_means_cluster(df1)
    
    assert len(x) == (len(df1.iloc[0,:]) - 2)
    

def test_gmm_cluster():
    
    #Testing gmm_cluster generate a cluster label for every dataframe point
    
    x = gmm_cluster(df1)
    
    assert len(x) == (len(df1.iloc[0,:]) - 2)


def test_images():
   
    #Testing images function generate the right number of matricies to be displayed 
    
    im = images(df1)
    
    assert len(im) == 6*(len(df1.iloc[0,:]) - 2)
    

def test_print_images():
    
    #Testing print_images function prints the right number of subplots
    
    pr = print_images(images(df1))
    m = images(df1)
    
    assert len(pr) == len(m)


