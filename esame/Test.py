# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:50:03 2019

@author: carlausss
"""

import numpy as np
import pandas as pd
import pytest

from Images import t_df, km, gmm, get_close_points


df1 = pd.DataFrame(np.random.randint(1,100,size=(100, 5)), columns=('X', 'Y','Alfa','Beta', 'Gamma'))
   
df2 = pd.DataFrame(np.random.randint(-100,0,size=(100, 5)), columns=('X', 'Y','Alfa','Beta', 'Gamma'))
   
df3 = pd.DataFrame(np.random.uniform(size = (100,5)), columns=('X', 'Y','Alfa','Beta', 'Gamma'))
    

#data = pd.read_excel("c:/Users/carlausss/Desktop/Prova.xlsx")
#data = pd.DataFrame(data)


def test_function_1():
    
    t_df(df1)
    print("\nCorrect DataFrame shape df1: \n\n",df1.head())
    
    #Testing that t_df function raises error with wrong DataFrames
    
    with pytest.raises(ValueError):
        
        t_df(df2)
        
        t_df(df3)
        

test_function_1()


def test_get_close_points():
    
    #Testing g_c_p raises ValueError for negative radius values
    
    with pytest.raises(ValueError):
        get_close_points(df1, radius = -2)


def test_km():
    
    #Testing km works correctly
    
    x = km(df1)
    
    assert len(x) == len(df1)
    

def test_gmm():
    
    #Testing gmm works correctly
    
    x = gmm(df1)
    
    assert len(x) == len(df1)

