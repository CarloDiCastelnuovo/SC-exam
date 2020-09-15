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


#data = pd.read_excel("c:/Users/carlausss/Desktop/S&C/Prova.xlsx")
#data = df.dropna()
#data = pd.DataFrame(data)

data = pd.DataFrame(np.random.randint(1,100,size=(100, 5)), columns=('X', 'Y','Alfa','Beta', 'Gamma'))
#data = pd.DataFrame(np.random.randint(-100,0,size=(100, 5)), columns=('X', 'Y','Alfa','Beta', 'Gamma'))


Num_of_parameters = (len(data.iloc[0,:]) - 2)

Columns_name = ['Alfa','Beta','Gamma']

def check_correct_coordinates(df):
    
    #Checking wheter the position data columns is composed by positive integer or not.
    
    min_x=min(df['X'])
    min_y=min(df['Y'])
    
    if min_x < 1:
        #raise ValueError('Wrong coordinates value or wrong columns coordinates position. The position data must be stored in the first two columns')
        return False
    
    if min_y < 1:
        #raise ValueError('Wrong coordinates value or wrong columns coordinates position. The position data must be stored in the first two columns')  
        return False
    else: 
        return True

def get_close_points(df, x, y, radius = 2):
    
    #The aim of this function is to collect the points within a radius, 
    #whose length can be modified in its definition, for each pixel of the image.
    #It returns the list of points within the radius.
    
    if radius < 0:
        raise ValueError('Radius value must be greater than zero')
    
    x_idx = df['X']
    y_idx = df['Y']
    dist_sqrd = (x_idx-x)**2 + (y_idx-y)**2
    to_take = np.sqrt(dist_sqrd)<=radius
    
    return df.loc[to_take]

                           
def k_means_cluster(df, nc = 2):
    
    #The function to collect the labels for K-Means clustering.
    #It returns a lists of labels ordered like the DataFrame which represents
    #membership in one of the clusters for every single pixel.
    
    kml = []
   
    km = cluster.KMeans(n_clusters = nc).fit(df[Columns_name[0]])   
    labels_km = km.labels_ 
    kml.append(labels_km)  
    
    km2 = cluster.KMeans(n_clusters = nc).fit(df[Columns_name[1]])   
    labels_km2 = km2.labels_ 
    kml.append(labels_km2)
    
    km3 = cluster.KMeans(n_clusters = nc).fit(df[Columns_name[2]])   
    labels_km3 = km3.labels_ 
    kml.append(labels_km3)
    
    if len(labels_km) != len(df):
        raise ValueError('Cluster data must be same length as dataframe')
    
    return kml


def gmm_cluster(df, nc = 2):

    #The function to collect the labels for GMM clustering.
    #It returns a lists of labels ordered like the DataFrame which represents
    #membership in one of the clusters for every single pixel.
    
    gmml = []
    
    gmm1 = GaussianMixture(n_components = nc).fit(df[Columns_name[0]])
    labels_gmm1 = gmm1.predict(df[Columns_name[0]])
    gmml.append(labels_gmm1)
    
    gmm2 = GaussianMixture(n_components = nc).fit(df[Columns_name[1]])
    labels_gmm2 = gmm2.predict(df[Columns_name[1]])
    gmml.append(labels_gmm2)
    
    gmm3 = GaussianMixture(n_components = nc).fit(df[Columns_name[2]])
    labels_gmm3 = gmm3.predict(df[Columns_name[2]])
    gmml.append(labels_gmm3)   
        
    if len(labels_gmm1) != len(df):
        raise ValueError('Cluster data must be same length as dataframe')
                         
    return gmml

#Create empty list to be filled with correct number of matricies that will be 
#generated in the for-cycle with the shape given by the maximun of x,y position data.


def fill_matricies_with_original_data(df):
    
    #It starts taking position data corresponding to 'X' and 'Y' columns in the first 
    #for-cycle, than for every parameter puts the value in correct matrix position.
    
    m = []  
    max_x=max(df['X'])+1
    max_y=max(df['Y'])+1 
        
    for i in range(Num_of_parameters):
   
        mat = np.empty((max_x,max_y))
        mat.fill(np.nan)            
        m.append(mat)
    
    for j in tqdm(range(len(data))):
        
        xp = df.loc[j, 'X'] 
        yp = df.loc[j, 'Y'] 
        
        m[0][int(xp), int(yp)] = df.loc[j,Columns_name[0]]
        m[1][int(xp), int(yp)] = df.loc[j,Columns_name[1]]
        m[2][int(xp), int(yp)] = df.loc[j,Columns_name[2]]
        
    return m
        
        
def fill_matricies_with_smoother_data(df):
    
    #After it creates a DataFrame with the values of the close points for the starting pixel,
    #it computes the mean of it and fill the matrices with the averaged value. This is done
    #for every point individually.
    
    m = []  
    max_x=max(data['X'])+1
    max_y=max(data['Y'])+1 
        
    for i in range(Num_of_parameters):
   
        mat = np.empty((max_x,max_y))
        mat.fill(np.nan)            
        m.append(mat)
    
    for j in tqdm(range(len(data))):
        
        xp = df.loc[j,'X'] 
        yp = df.loc[j,'Y']
    
        parda = get_close_points(data, xp, yp, radius = 2)
        parda = pd.DataFrame(parda)
        
        cp = parda[Columns_name[0]]
        cp = cp.mean()
        m[0][int(xp), int(yp)] = cp   
    
        cp = parda[Columns_name[1]]
        cp = cp.mean()
        m[1][int(xp), int(yp)] = cp   
    
        cp = parda[Columns_name[2]]
        cp = cp.mean()
        m[2][int(xp), int(yp)] = cp   
    
    return m
    
def fill_matricies_with_kMeansCluster_data(df):

    #Same procedure as the previous function to collect close points.
    #Then the cluster algorithm is iterated both on the original data and on the DF of the close points. 
    #Finally, for each parameter, two matrices representing the results of the cluster are filled.
    
    m = []  
    max_x=max(data['X'])+1
    max_y=max(data['Y'])+1  
        
    for i in range(2*Num_of_parameters):
   
        mat = np.empty((max_x,max_y))
        mat.fill(np.nan)            
        m.append(mat)
    
    for j in tqdm(range(len(data))):
        
        xp = df.loc[j,'X'] 
        yp = df.loc[j,'Y']
    
        parda = get_close_points(data, xp, yp, radius = 2)
        parda = pd.DataFrame(parda)
    
    kma = k_means_cluster(data, nc=2)   
    kma_av = k_means_cluster(parda, nc=2)  

    kma = pd.DataFrame(kma)   
    kma_av = pd.DataFrame(kma_av)
    
    for l in tqdm(range(len(data))):
                
        xp = df.loc[l,'X'] 
        yp = df.loc[l,'Y']
        
        for n in range(Num_of_parameters):

            m[n][int(xp), int(yp)] = kma.iloc[0+n,l]     
            m[Num_of_parameters+n][int(xp), int(yp)] = kma_av.iloc[0+n,l]      
                 
    return m
   
def fill_matricies_with_gmmCluster_data(df):
    
    #Same procedure as the previous function to collect close points.
    #Then the cluster algorithm is iterated both on the original data and on the DF of the close points. 
    #Finally, for each parameter, two matrices representing the results of the cluster are filled.
    
    m = []  
    max_x=max(data['X'])+1
    max_y=max(data['Y'])+1 
        
    for i in range(2*Num_of_parameters):
   
        mat = np.empty((max_x,max_y))
        mat.fill(np.nan)            
        m.append(mat)

    for j in tqdm(range(len(data))):
        
        xp = df.loc[j,'X'] 
        yp = df.loc[j,'Y']
    
        parda = get_close_points(data, xp, yp, radius = 2)
        parda = pd.DataFrame(parda)
     
    gmma = gmm_cluster(data, nc=2)
    gmma_av = gmm_cluster(parda, nc=2)
    
    gmma = pd.DataFrame(gmma)
    gmma_av = pd.DataFrame(gmma_av)
        
    for l in tqdm(range(len(data))):
                
        xp = df.loc[l,'X'] 
        yp = df.loc[l,'Y']
        
        for n in range(Num_of_parameters):

            m[n][int(xp), int(yp)] = gmma.iloc[0+n,l]

            m[Num_of_parameters+n][int(xp), int(yp)] = gmma_av.iloc[0+n,l]                    
    return m    

def print_original_images(m):           

    #Generates subplots for matricies that show the original values for every parameters
    
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







