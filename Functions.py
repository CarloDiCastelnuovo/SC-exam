# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 18:09:05 2020

@author: carlausss
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster
from sklearn.mixture import GaussianMixture

from tqdm import tqdm


def correct_df(df):
    
    min_x=min(df.iloc[:,0])
    min_y=min(df.iloc[:,1])
    
    if min_x < 1:
        raise ValueError('Wrong coordinates value')
    
    if min_y < 1:
        raise ValueError('Wrong coordinates value')  


def get_close_points(df, x, y, radius = 2):

    correct_df(df)
    
    if radius < 0:
        raise ValueError('Radius value must be greater than zero')
    
    x_idx = df.iloc[:, 0]
    y_idx = df.iloc[:, 1]
    dist_sqrd = (x_idx-x)**2 + (y_idx-y)**2
    to_take = np.sqrt(dist_sqrd)<=radius
    
    return df.loc[to_take]

                           
def km(df, nc = 2 ):
    
    correct_df(df)

    kml = []

    col = (len(df.iloc[0,:]) - 2)    
    for i in range(col):
        
        km = cluster.KMeans(n_clusters = nc).fit(df.iloc[:, 2+i:3+i])   
        labels_km = km.labels_ 
        kml.append(labels_km)
    
    if len(labels_km) != len(df):
        raise ValueError('Cluster df must be same length as DataFrame')
    
    return kml


def gmm(df, nc = 2):
    
    correct_df(df)

    gmml = []
    
    col = (len(df.iloc[0,:]) - 2)
    for i in range(col):
 
        gmm = GaussianMixture(n_components = nc).fit(df.iloc[:, 2+i:3+i])
        labels_gmm = gmm.predict(df.iloc[:, 2+i:3+i])
        gmml.append(labels_gmm)   
        
    if len(labels_gmm) != len(df):
        raise ValueError('Cluster df must be same length as DataFrame')
                         
    return gmml


def images(df):

    correct_df(df)

    m = []  
    max_x=max(df.iloc[:,0])+1
    max_y=max(df.iloc[:,1])+1
    
    col = (len(df.iloc[0,:]) - 2)

    for i in range(6*col):
        mat = np.empty((max_x,max_y))
        mat.fill(np.nan)        
        
        m.append(mat)
        
    for j in tqdm(range(len(df))):
        
        xp = df.iloc[j,0] 
        yp = df.iloc[j,1] 
        
        for i in range(col):
        
            m[i][int(xp), int(yp)] = df.iloc[j,2+i]
        
    
        parda = get_close_points(df, xp, yp, radius = 2)
        parda = pd.DataFrame(parda)
        
        for i in range(col):
        
            cp = parda.iloc[:,2+i]
            cp = cp.mean()
    
            m[col+i][int(xp), int(yp)] = cp   
       
    kma = km(df, nc=2)   
    gmma = gmm(df, nc=2)
    
    kma_av = km(parda, nc=1)  
    gmma_av = gmm(parda, nc=1)
    
    kma = pd.DataFrame(kma)
    
    kma_av = pd.DataFrame(kma_av)
    
    gmma = pd.DataFrame(gmma)
    
    gmma_av = pd.DataFrame(gmma_av)
        
    for l in tqdm(range(len(df))):
        
        xp = df.iloc[l,0] 
        yp = df.iloc[l,1] 
        
        for n in range(col):

            m[2*col+n][int(xp), int(yp)] = kma.iloc[0+n,l]     

            m[3*col+n][int(xp), int(yp)] = gmma.iloc[0+n,l]
        
            m[4*col+n][int(xp), int(yp)] = kma_av.iloc[0+n,l]      

            m[5*col+n][int(xp), int(yp)] = gmma_av.iloc[0+n,l]
        
        if len(m) != 6*col:
            raise ValueError('The number of matrices is not correct')
        
    return m
   

def print_images(m, df):           
    
    fig = plt.figure(figsize=(15, 25))
    ax = []
    
    col = (len(df.iloc[0,:]) - 2)

    for i in range(6*col):
    
        ax.append( fig.add_subplot(8, col, i+1) )
    
        ax[-1].set_xlabel('X')
        ax[-1].set_ylabel('Y')    
    
        plt.imshow(m[i])
        
    for i in range(col):    
        
        ax[i].set_title("Values")
        ax[col+i].set_title("Averaged Values")
        ax[2*col+i].set_title("KM Cluster")
        ax[3*col+i].set_title("GMM Cluster")
        ax[4*col+i].set_title("KM Cluster on Averaged Values")
        ax[5*col+i].set_title("GMM Cluster on Averaged Values")   
        
    if len(ax) != images(df):
        raise ValueError('The number of subplots is not correct')

