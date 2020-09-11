# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 15:51:33 2020

@author: carlausss
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster
from sklearn.mixture import GaussianMixture

from tqdm import tqdm

    #Example of a correct DataFrame
data = pd.DataFrame(np.random.randint(1,100,size=(100, 5)), columns=('X', 'Y','Alfa','Beta', 'Gamma'))

    #Example of an incorrect DataFrame
#data = pd.DataFrame(np.random.randint(-100,0,size=(100, 5)), columns=('X', 'Y','Alfa','Beta', 'Gamma'))

    #Number of parameters
Num_of_parameters = (len(data.iloc[0,:]) - 2)

def check_correct_coordinates(df):
    
    #Checking wheter the position data columns is composed by positive integer or not.
    
    min_x=min(df.loc[:,'X'])
    min_y=min(df.loc[:,'Y'])
    
    if min_x < 1:
        return False
    
    if min_y < 1:
        return False
    else: 
        return True

def get_close_points(df, x, y, radius = 2):
    
    #The aim of this function is to collect the points within a radius, 
    #whose length can be modified in its definition, for each pixel of the image.
    #It returns the list of points within the radius.
    
    #The arguments required arethe main DataFrame, the starting points and value of radius.
    
    if radius < 0:
        raise ValueError('Radius value must be greater than zero')
    
    x_idx = data.loc[:, 'X']
    y_idx = data.loc[:, 'Y']
    dist_sqrd = (x_idx-x)**2 + (y_idx-y)**2
    to_take = np.sqrt(dist_sqrd)<=radius
    
    return data.loc[to_take]

                           
def k_means_cluster(df, nc = 2 ):
    
    #The function to collect the labels for K-Means clustering.
    #It returns a lists of labels ordered like the DataFrame which represents
    #membership in one of the clusters for every single pixel.
 
    #The arguments required are tha DataFrame (df) and the number of cluster
    
    kml = []
    
    for i in range(Num_of_parameters):
        
        km = cluster.KMeans(n_clusters = nc).fit(data.iloc[:, 2+i:3+i])   
        labels_km = km.labels_ 
        kml.append(labels_km)
    
    if len(labels_km) != len(data):
        raise ValueError('Cluster data must be same length as dataframe')
    
    return kml


def gmm_cluster(df, nc = 2):

    #The function to collect the labels for GMM clustering.
    #It returns a lists of labels ordered like the DataFrame which represents
    #membership in one of the clusters for every single pixel.
    
    #The arguments required are tha DataFrame (df) and the number of cluster
    
    gmml = []
    
    for i in range(Num_of_parameters):
 
        gmm = GaussianMixture(n_components = nc).fit(data.iloc[:, 2+i:3+i])
        labels_gmm = gmm.predict(data.iloc[:, 2+i:3+i])
        gmml.append(labels_gmm)   
        
    if len(labels_gmm) != len(data):
        raise ValueError('Cluster data must be same length as dataframe')
                         
    return gmml

def fill_matricies_with_original_data(df):
    
    #The first step is to create a list of empty matrices, one for each parameter, of the correct size.
    #Than it starts taking position data corresponding to 'X' and 'Y' columns in the first 
    #for-cycle, finally for every parameter puts the value in correct matrix position.
    
    #The only argument required is tha DataFrame (df)
    
    m = []  
    max_x=max(data.loc[:,'X'])+1
    max_y=max(data.loc[:,'Y'])+1 
        
    for i in range(Num_of_parameters):
   
        mat = np.empty((max_x,max_y))
        mat.fill(np.nan)            
        m.append(mat)
    
    for j in tqdm(range(len(data))):
        
        xp = data.loc[j, 'X'] 
        yp = data.loc[j, 'Y'] 
        
        m[0][int(xp), int(yp)] = data.loc[j,'Alfa']
        m[1][int(xp), int(yp)] = data.loc[j,'Beta']
        m[2][int(xp), int(yp)] = data.loc[j,'Gamma']
        
    return m
        
        
def fill_matricies_with_smoother_data(df):
    
    #The first step is to create a list of empty matrices, one for each parameter, of the correct size.
    #After it creates a DataFrame with the values of the close points for the starting pixel,
    #it computes the mean of it and fill the matrices with the averaged value. This is done
    #for every point individually.
    
    #The only argument required is tha DataFrame (df)

    m = []  
    max_x=max(data.loc[:,'X'])+1
    max_y=max(data.loc[:,'Y'])+1 
        
    for i in range(Num_of_parameters):
   
        mat = np.empty((max_x,max_y))
        mat.fill(np.nan)            
        m.append(mat)
    
    for j in tqdm(range(len(data))):
        
        xp = data.loc[j,'X'] 
        yp = data.loc[j,'Y']
    
        parda = get_close_points(data, xp, yp, radius = 2)
        parda = pd.DataFrame(parda)
        
        cp = parda.loc[:,'Alfa']
        cp = cp.mean()
        m[0][int(xp), int(yp)] = cp   
    
        cp = parda.loc[:,'Beta']
        cp = cp.mean()
        m[1][int(xp), int(yp)] = cp   
    
        cp = parda.loc[:,'Gamma']
        cp = cp.mean()
        m[2][int(xp), int(yp)] = cp   
    
    return m
    
def fill_matricies_with_kMeansCluster_data(df):

    #The first step is to create a list of empty matrices, one for each parameter, of the correct size.
    #Same procedure as the previous function to collect close points.
    #Then the cluster algorithm is iterated both on the original data and on the DF of the close points. 
    #Finally, for each parameter, two matrices representing the results of the cluster are filled.
    
    #The only argument required is tha DataFrame (df)
    
    m = []  
    max_x=max(data.loc[:,'X'])+1
    max_y=max(data.loc[:,'Y'])+1 
        
    for i in range(2*Num_of_parameters):
   
        mat = np.empty((max_x,max_y))
        mat.fill(np.nan)            
        m.append(mat)
    
    for j in tqdm(range(len(data))):
        
        xp = data.loc[j,'X'] 
        yp = data.loc[j,'Y']
    
        parda = get_close_points(data, xp, yp, radius = 2)
        parda = pd.DataFrame(parda)
    
    kma = k_means_cluster(data, nc=2)   
    kma_av = k_means_cluster(parda, nc=2)  

    kma = pd.DataFrame(kma)   
    kma_av = pd.DataFrame(kma_av)
    
    for l in tqdm(range(len(data))):
                
        xp = data.loc[l,'X'] 
        yp = data.loc[l,'Y']
        
        for n in range(Num_of_parameters):

            m[n][int(xp), int(yp)] = kma.iloc[0+n,l]     
            m[Num_of_parameters+n][int(xp), int(yp)] = kma_av.iloc[0+n,l]      
                 
    return m
   
def fill_matricies_with_gmmCluster_data(df):
    
    #The first step is to create a list of empty matrices, one for each parameter, of the correct size.
    #Same procedure as the previous function to collect close points.
    #Then the cluster algorithm is iterated both on the original data and on the DF of the close points. 
    #Finally, for each parameter, two matrices representing the results of the cluster are filled.
    
    #The only argument required is tha DataFrame (df)
    
    m = []  
    max_x=max(data.loc[:,'X'])+1
    max_y=max(data.loc[:,'Y'])+1 
        
    for i in range(2*Num_of_parameters):
   
        mat = np.empty((max_x,max_y))
        mat.fill(np.nan)            
        m.append(mat)

    for j in tqdm(range(len(data))):
        
        xp = data.loc[j,'X'] 
        yp = data.loc[j,'Y']
    
        parda = get_close_points(data, xp, yp, radius = 2)
        parda = pd.DataFrame(parda)
     
    gmma = gmm_cluster(data, nc=2)
    gmma_av = gmm_cluster(parda, nc=2)
    
    gmma = pd.DataFrame(gmma)
    gmma_av = pd.DataFrame(gmma_av)
        
    for l in tqdm(range(len(data))):
                
        xp = data.loc[l,'X'] 
        yp = data.loc[l,'Y']
        
        for n in range(Num_of_parameters):

            m[n][int(xp), int(yp)] = gmma.iloc[0+n,l]

            m[Num_of_parameters+n][int(xp), int(yp)] = gmma_av.iloc[0+n,l]                    
    return m    

def print_original_images(m):           

    #Generates subplots for matricies that show the original values for every parameters.
    #The only argument required is the list of the corresponding matricies.
    
    fig = plt.figure(figsize=(15, 25))
    ax = []

    for i in range(Num_of_parameters):
    
        ax.append( fig.add_subplot(1, Num_of_parameters, i+1) )
    
        ax[-1].set_xlabel('X')
        ax[-1].set_ylabel('Y')    
    
        plt.imshow(m[i])
        
    ax[0].set_title("Alfa Values")
    ax[1].set_title("Beta Values")
    ax[2].set_title("Gamma Values")
        
    return ax    
        
def print_smoother_images(m):           

    #Generates subplots for matricies that show the avaraged values for every parameters
    #The only argument required is the list of the corresponding matricies.
    
    fig = plt.figure(figsize=(15, 25))
    ax = []

    for i in range(Num_of_parameters):
    
        ax.append( fig.add_subplot(1, Num_of_parameters, i+1) )
    
        ax[-1].set_xlabel('X')
        ax[-1].set_ylabel('Y')    
    
        plt.imshow(m[i])
        
    ax[0].set_title("Averaged Alfa Values")        
    ax[1].set_title("Averaged Beta Values")       
    ax[2].set_title("Averaged Gamma Values")       
    
    return ax

def print_kMeansCluster_images(m):          

    #Generates subplots for matricies that show the K-Means clustering results on original data    
    #The only argument required is the list of the corresponding matricies.
    
    fig = plt.figure(figsize=(15, 25))
    ax = []

    for i in range(Num_of_parameters):
    
        ax.append( fig.add_subplot(1, Num_of_parameters, i+1) )
    
        ax[-1].set_xlabel('X')
        ax[-1].set_ylabel('Y')    
    
        plt.imshow(m[i])
        
    ax[0].set_title("K-Means Cluster Alfa results")        
    ax[1].set_title("K-Means Cluster Beta results")        
    ax[2].set_title("K-Means Cluster Gamma results")  
    
    return ax 

def print_kMeansCluster_AveragedImages(m):
    
    #Generates subplots for matricies that show the K-Means clustering results on averaged data    
    #The only argument required is the list of the corresponding matricies.
    
    fig = plt.figure(figsize=(15, 25))
    ax = []

    for i in range(Num_of_parameters):
    
        ax.append( fig.add_subplot(1, Num_of_parameters, i+1) )
    
        ax[-1].set_xlabel('X')
        ax[-1].set_ylabel('Y')    
    
        plt.imshow(m[i])
    
    ax[0].set_title("K-Means Cluster Averaged Alfa results")        
    ax[1].set_title("K-Means Cluster Averaged Beta results")        
    ax[2].set_title("K-Means Cluster Averaged Gamma results")
    
    return ax
        
def print_gmmCluster_images(m):           

    #Generates subplots for matricies that show the GMM clustering results on original data    
    #The only argument required is the list of the corresponding matricies.
    
    fig = plt.figure(figsize=(15, 25))
    ax = []

    for i in range(Num_of_parameters):
    
        ax.append( fig.add_subplot(1, Num_of_parameters, i+1) )
        
        ax[-1].set_xlabel('X')
        ax[-1].set_ylabel('Y')    
    
        plt.imshow(m[i])

    ax[0].set_title("GMM Cluster Alfa results")        
    ax[1].set_title("GMM Cluster Beta results")        
    ax[2].set_title("GMM Cluster Gamma results") 

    return ax   

def print_gmmCluster_AveragedImages(m):
    
    #Generates subplots for matricies that show the GMM clustering results on averaged data    
    #The only argument required is the list of the corresponding matricies.
    
    fig = plt.figure(figsize=(15, 25))
    ax = []

    for i in range(Num_of_parameters):
    
        ax.append( fig.add_subplot(1, Num_of_parameters, i+1) )
    
        ax[-1].set_xlabel('X')
        ax[-1].set_ylabel('Y')    
    
        plt.imshow(m[i])
    
    ax[0].set_title("GMM Cluster Averaged Alfa results")        
    ax[1].set_title("GMM Cluster Averaged Beta results")        
    ax[2].set_title("GMM Cluster Averaged Gamma results")

    return ax
    
check_correct_coordinates(data)
print_original_images(fill_matricies_with_original_data(data))
print_smoother_images(fill_matricies_with_smoother_data(data))
print_kMeansCluster_images(fill_matricies_with_kMeansCluster_data(data))
print_gmmCluster_images(fill_matricies_with_gmmCluster_data(data))
print_kMeansCluster_AveragedImages(fill_matricies_with_kMeansCluster_data(data))
print_gmmCluster_AveragedImages(fill_matricies_with_gmmCluster_data(data))







