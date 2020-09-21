# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 12:05:29 2020

@author: carlausss
"""


import numpy as np
import pandas as pd

from Functions import check_correct_coordinates, get_close_points, k_means_cluster, gmm_cluster

# Generate 3 different DataFrame to test the functions, df1 has correct shape df2 and df3 have not

df1 = pd.DataFrame(np.random.randint(1,100,size=(100, 5)), columns=('X', 'Y','Alfa','Beta', 'Gamma'))
   
df2 = pd.DataFrame(np.random.randint(-100,0,size=(100, 5)), columns=('X', 'Y','Alfa','Beta', 'Gamma'))
   
df3 = pd.DataFrame(np.random.uniform(size = (100,5)), columns=('X', 'Y','Alfa','Beta', 'Gamma'))
    
print("\nCorrect DataFrame shape df1: \n\n",df1.head())
    
def test_check_correct_coordinates():
    
    # Testing that the check function return True with correct DF and False with wrong DF
    
    check1 = check_correct_coordinates(df1)
    assert check1 == True
    
    check2 = check_correct_coordinates(df2)
    check3 = check_correct_coordinates(df3)
    assert check2 == False
    assert check3 == False
    

def test_get_close_points():
    
    # Testing get_close_points function gives correct number of close points for ad hoc DataFrame:
    # the first 3 points of dfcp (0,1)(1,0)(1,1) are within get_close_points radius, the last (4,5) is not. 
    
    dfcp = pd.DataFrame({'X' : [0,1,1,4], 'Y' : [1,0,1,5], 'test_value' : [9, 10, 11, 12]})
    
    l = []
    
    for i in range(len(dfcp)):    
        x = dfcp.iloc[0:1, 0:1]
        y = dfcp.iloc[0:1, 1:2]
        
        r2 = get_close_points(dfcp, x, y, radius = 2)
        l.append(r2)
        
        r6 = get_close_points(dfcp, x, y, radius = 6)
        l.append(r6)
        
    assert len(l[0]) == 3
    assert len(l[1]) == 4
        
    
def test_k_means_cluster():
    
    # Testing k_means_cluster generate a cluster label for every dataframe point.
    # The labels are integer number starting from 0, thus for nc = 2 we aspect labels made by 0 or 1
    # for nc = 4 we aspect labels made by (0,1,2,3)
    
    x = k_means_cluster(df1, 'test_value', 2)

    assert len(x) == len(df1)
    
    assert np.max(x) <= 1
    
    x = k_means_cluster(df1, 'test_value', 4)
    
    assert np.min(x) == 0
    assert np.max(x) == 3
    

def test_gmm_cluster():
    
    # Testing gmm_cluster generate a cluster label for every dataframe point
    # The labels are integer number starting from 0, thus for nc = 2 we aspect labels made by 0 or 1
    # for nc = 4 we aspect labels made by (0,1,2,3)
    
    x = gmm_cluster(df1,'test_value', 2)
   
    assert len(x) == len(df1)
       
    assert np.max(x) <= 1
    
    x = gmm_cluster(df1,'test_value', 4)
    
    assert np.max(x) == 3
    assert np.min(x) == 0
