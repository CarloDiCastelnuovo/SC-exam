# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:29:39 2020

@author: carlausss
"""


import pandas as pd
import numpy as np
from sklearn import cluster
from sklearn.mixture import GaussianMixture
from tqdm import tqdm


def check_correct_coordinates(df):
    
    #Checking wheter the position data columns is composed by positive integer or not.
    
    min_x=min(df['X'])
    min_y=min(df['Y'])
    
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
    
    if radius < 0:
        raise ValueError('Radius value must be greater than zero')
        
    x_idx = df['X']
    y_idx = df['Y']
    dist_sqrd = (x_idx-x)**2 + (y_idx-y)**2
    to_take = np.sqrt(dist_sqrd)
    
    return df.loc[to_take]


def k_means_cluster(df, nc, col_name):
    
    #The function to Num_of_parameterslect the labels for K-Means clustering.
    #It returns a lists of labels ordered like the DataFrame which represents
    #membership in one of the clusters for every single pixel.
    
    kml = []    
    tofit = df[col_name].values
    tofit = tofit.reshape(2, -1)
    km = cluster.KMeans(n_clusters = nc).fit(tofit)   
    #km = cluster.KMeans(n_clusters = nc).fit(df.iloc[:,2:3])
    labels_km = km.labels_ 
    kml.append(labels_km)
    kml = pd.DataFrame(kml, columns = [col_name,' K-Means Labels'])
    
    if len(kml) != len(df):
        raise ValueError('Cluster data must be same length as dataframe')
    
    return kml


def gmm_cluster(df, nc, col_name):

    #The function to Num_of_parameterslect the labels for GMM clustering.
    #It returns a lists of labels ordered like the DataFrame which represents
    #membership in one of the clusters for every single pixel.
    
    gmml = []
    gmm = GaussianMixture(n_components = nc).fit(df[col_name].values)
    labels_gmm = gmm.predict(df[col_name])
    gmml.append(labels_gmm)   
    gmml = pd.DataFrame(gmml, columns = [col_name,' GMM Labels'])
        
    if len(gmml) != len(df):
        raise ValueError('Cluster data must be same length as dataframe')
                         
    return gmml

def fill_matricies_with_original_data(df, col_name):
    
    #The function creates a empty matrix and it fills it taking one by one position data and parameter value.
    
    m = []  
    max_x=max(df['X'])+1
    max_y=max(df['Y'])+1 
        
    mat = np.empty((max_x,max_y))
    mat.fill(np.nan)            
    m.append(mat)
    
    for j in tqdm(range(len(df))):
        
        xp = df.loc[j, 'X'] 
        yp = df.loc[j, 'Y'] 
        
        m[0][int(xp), int(yp)] = df.loc[j, col_name]
        
    return m


def fill_matricies_with_smoother_data(df, col_name):
    
    #After it creates a DataFrame with the values of the close points for the starting pixel,
    #it computes the mean of it and fill the matrices with the averaged value. This is done
    #for every point individually.
    
    m = []  
    max_x=max(df['X'])+1
    max_y=max(df['Y'])+1 
   
    mat = np.empty((max_x,max_y))
    mat.fill(np.nan)            
    m.append(mat)
    
    for j in tqdm(range(len(df))):
        
        xp = df.loc[j,'X'] 
        yp = df.loc[j,'Y']
        parda = get_close_points(df, xp, yp, radius = 2)
        cp = parda[col_name]
        cp = cp.mean()
        m[0][int(xp), int(yp)] = cp 
    
    return m


def fill_matricies_with_kMeansCluster_data(df, kml):

    #Same procedure as the previous function to collect close points.
    #Then the cluster algorithm is iterated both on the original data and on the DF of the close points. 
    #Finally, for each parameter, two matrices representing the results of the cluster are filled.
    
    m = []  
    max_x=max(df['X'])+1
    max_y=max(df['Y'])+1     
    mat = np.empty((max_x,max_y))
    mat.fill(np.nan)            
    m.append(mat)  
        
    for l in tqdm(range(len(df))):
                
        xp = df.loc[l,'X'] 
        yp = df.loc[l,'Y']
        m[0][int(xp), int(yp)] = kml.iloc[0,l]     
                 
    return m


def fill_matricies_with_kMeansCluster_AveragedData(df, kml):

    #Same procedure as the previous function to collect close points.
    #Then the cluster algorithm is iterated both on the original data and on the DF of the close points. 
    #Finally, for each parameter, two matrices representing the results of the cluster are filled.
    
    m = []  
    max_x=max(df['X'])+1
    max_y=max(df['Y'])+1     
    mat = np.empty((max_x,max_y))
    mat.fill(np.nan)            
    m.append(mat)
    
    for j in tqdm(range(len(df))):
        
        xp = df.loc[j,'X'] 
        yp = df.loc[j,'Y']
        parda = get_close_points(df, xp, yp, radius = 2)
    
    kma_av = k_means_cluster(parda)  
    
    for l in tqdm(range(len(df))):
                
        xp = df.loc[l,'X'] 
        yp = df.loc[l,'Y']
        m[0][int(xp), int(yp)] = kma_av.iloc[0,l]      
                 
    return m


def fill_matricies_with_gmmCluster_data(df, gmml):
    
    #Same procedure as the previous function to collect close points.
    #Then the cluster algorithm is iterated both on the original data and on the DF of the close points. 
    #Finally, for each parameter, two matrices representing the results of the cluster are filled.
    
    m = []  
    max_x=max(df['X'])+1
    max_y=max(df['Y'])+1     
    mat = np.empty((max_x,max_y))
    mat.fill(np.nan)            
    m.append(mat)
    
    for l in tqdm(range(len(df))):
                
        xp = df.loc[l,'X'] 
        yp = df.loc[l,'Y']
        m[0][int(xp), int(yp)] = gmml.iloc[0,l]      
                 
    return m


def fill_matricies_with_gmmCluster_AveragedData(df, gmml):
    
    #Same procedure as the previous function to collect close points.
    #Then the cluster algorithm is iterated both on the original data and on the DF of the close points. 
    #Finally, for each parameter, two matrices representing the results of the cluster are filled.
    
    m = []  
    max_x=max(df['X'])+1
    max_y=max(df['Y'])+1     
    mat = np.empty((max_x,max_y))
    mat.fill(np.nan)            
    m.append(mat)
    
    for j in tqdm(range(len(df))):
        
        xp = df.loc[j,'X'] 
        yp = df.loc[j,'Y']
        parda = get_close_points(df, xp, yp, radius = 2)
       
    gmm_av = gmm_cluster(parda, nc=2)  
    
    for l in tqdm(range(len(df))):
                
        xp = df.loc[l,'X'] 
        yp = df.loc[l,'Y']
        m[0][int(xp), int(yp)] = gmm_av.iloc[0,l]      
                 
    return m


