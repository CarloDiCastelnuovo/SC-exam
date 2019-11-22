# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:50:03 2019

@author: carlausss
"""

import numpy as np
import pandas as pd
import pytest

df1 = pd.DataFrame(np.random.randint(1,100,size=(100, 5)), columns=('X', 'Y','Alfa','Beta', 'Gamma'))
   
df2 = pd.DataFrame(np.random.randint(-100,0,size=(100, 5)), columns=('X', 'Y','Alfa','Beta', 'Gamma'))
   
df3 = pd.DataFrame(np.random.uniform(size = (100,5)), columns=('X', 'Y','Alfa','Beta', 'Gamma'))
    

def t_df(df):
    
    min_x=min(df.iloc[:,0])
    min_y=min(df.iloc[:,1])
    
    if min_x < 1:
        raise ValueError('Wrong coordinates value ', min_x,' is not correct')
    
    if min_y < 1:
        raise ValueError('Wrong coordinates value ', min_y,' is not correct')  
    
    for i in range(len(df)):
        
        if df.iloc[i,:] == np.nan:
            raise ValueError ('All the columns must have the same lenght')
        elif df.iloc[:,i] == np.nan:
            raise ValueError ('All the columns must have the same lenght')
    
def test_function_1():
    
    t_df(df1)
    print("DataFrame df1: \n",df1.head())
    
def test_function_2():
    
    t_df(df2)

def test_function_3():
    
    t_df(df3)
    
test_function_1()
test_function_4()


