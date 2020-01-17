# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:50:03 2019

@author: carlausss
"""

import numpy as np
import pandas as pd
import pytest

from esame.Functions import correct_df, km, gmm, get_close_points, images, print_images


df1 = pd.DataFrame(np.random.randint(1,100,size=(100, 5)), columns=('X', 'Y','Alfa','Beta', 'Gamma'))
   
df2 = pd.DataFrame(np.random.randint(-100,0,size=(100, 5)), columns=('X', 'Y','Alfa','Beta', 'Gamma'))
   
df3 = pd.DataFrame(np.random.uniform(size = (100,5)), columns=('X', 'Y','Alfa','Beta', 'Gamma'))
    

def test_df():
    
    correct_df(df1)
    print("\nCorrect DataFrame shape df1: \n\n",df1.head())
    
    #Testing that correct_df function raises error with wrong DataFrames
    
    with pytest.raises(ValueError):
        
        correct_df(df2)
        
        correct_df(df3)
        

test_df()


def test_get_close_points():
    
    #Testing g_c_p raises ValueError for negative radius values
    
    with pytest.raises(ValueError):
        
        x = df1.iloc[:, 0]
        y = df1.iloc[:, 1]
        get_close_points(df1, x, y, radius = -2)

    print("\ngcp it's ok")


def test_km():
    
    #Testing km works correctly
    
    x = km(df1)
    
    assert len(x) == (len(df1.iloc[0,:]) - 2)
    

def test_gmm():
    
    #Testing gmm works correctly
    
    x = gmm(df1)
    
    assert len(x) == (len(df1.iloc[0,:]) - 2)


def test_images():
   
    #Testing images works correctly
    
    im = images(df1)
    
    assert len(im) == 6*(len(df1.iloc[0,:]) - 2)
    

def test_print():
    
    #Testing print works correctly
    
    pr = print_images(images(df1), df1)
    
    assert len(pr) == len(images(df1))


