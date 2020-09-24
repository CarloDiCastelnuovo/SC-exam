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
import matplotlib.pyplot as plt


def check_correct_coordinates(df):
   
    """
    Checking whether the position data columns is composed by positive integer or not.
    
    Args:
        df: the DataFrame to be checked 
        
    Returns:
        Bool: True if the df is correct, False if it is not
        
    """
    min_x=min(df['X'])
    min_y=min(df['Y'])
    
    if min_x < 1:
        return False
    
    if min_y < 1:
        return False
    else: 
        return True
    
    
def get_close_points(df, x, y, radius = 2):
   
    """
    The aim of this function is to collect the points within a radius, 
    whose length can be modified in its definition, for each pixel of the image.
    
    Args:
        df: the DataFrame to be analyzed
        x, y: the coordinates of the point from which the function will start to collect close points
        radius: the length of the that defines the close points 
    
    Return:
        to_take: a DataFrame containing neighboring points of (x,y) 
         
    """
    if radius < 0:
        raise ValueError('Radius value must be greater than zero')
        
    x_idx = df.loc[:, 'X']
    y_idx = df.loc[:, 'Y']
    dist_sqrd = (x_idx-x)**2 + (y_idx-y)**2
    to_take = np.sqrt(dist_sqrd) <= radius
    cp_df = df[to_take]
    cp_df = pd.DataFrame(cp_df)
    
    return cp_df


def k_means_cluster(df, a, b, nc):
    
    """
    The function to collect the labels for K-Means clustering, coded by integer numbers

    Args:
        df: the DataFrame to be analyzed
        a,b: numerical indices of the column to be clustered
        nc: the number of clusters to be generated
        
    Return:
        kml: a DataFrame containing the labels for each point ordered like the original df

    """
    
    kml = []    
    tofit = df.iloc[:, a:b]
    km = cluster.KMeans(n_clusters = nc).fit(tofit)   
    labels_km = km.labels_ 
    kml.append(labels_km)
    kml = pd.DataFrame(kml)
    
    if len(kml.stack()) != len(df):
        raise ValueError('Cluster data must be same length as dataframe')
        
    return kml


def gmm_cluster(df, a, b, nc):

    """
    The function to collect the labels for GMM clustering, coded by integer numbers

    Args:
        df: the DataFrame to be analyzed
        a,b: numerical indices of the column to be clustered
        nc: the number of clusters to be generated
        
    Return:
        gmml: a DataFrame containing the labels for each point ordered like the original df

    """
    gmml = []
    tofit = df.iloc[:, a:b].values
    gmm = GaussianMixture(n_components = nc).fit(tofit)
    labels_gmm = gmm.predict(tofit)
    gmml.append(labels_gmm)   
    gmml = pd.DataFrame(gmml)

    if len(gmml.stack()) != len(df):
        raise ValueError('Cluster data must be same length as dataframe')
                         
    return gmml

def fill_matricies_with_original_data(df, col_name):
    
    """
    Creates an empty matrix and fills it taking one by one position data and parameter value.
    
    Args:
        df: the DataFrame to be analyzed
        a,b: numerical indices of the column to be clustered
    
    Return:
        mat: the filled matrix
    """
    max_x=max(df['X'])+1
    max_y=max(df['Y'])+1      
    mat = np.empty((max_x,max_y))
    mat.fill(np.nan)            
    
    for j in tqdm(range(len(df))):
        
        xp = df.loc[j, 'X'] 
        yp = df.loc[j, 'Y'] 
        
        mat[int(xp), int(yp)] = df.loc[j, col_name]
        
    return mat


def fill_matricies_with_averaged_data(df, col_name):
    
    """
    Creates an empty matrix and taking one by one position data replaces the 
    value of each pixel with the average of the neighboring ones
    
    Args:
        df: the DataFrame to be analyzed
        col_name: the name of the column to be smoothered
    
    Return:
        mat: the filled smooth matrix
    """
    
    max_x=max(df['X'])+1
    max_y=max(df['Y'])+1      
    mat = np.empty((max_x,max_y))
    mat.fill(np.nan)            
    
    
    for j in tqdm(range(len(df))):
        
        xp = df.loc[j,'X'] 
        yp = df.loc[j,'Y']
        parda = get_close_points(df, xp, yp, radius = 2)
        cp = parda[col_name].values.mean()
        mat[int(xp), int(yp)] = cp 
    
    return mat


def fill_matricies_with_kMeansCluster_data(df, a, b, nc):

    """
    Creates an empty matrix and fills it with cluster labels 
    returned by k_means_cluster function
    
    Args:
        df: the DataFrame to be analyzed
        a,b: numerical indices of the column to be clustered
        nc: the number of clusters to be generated
    
    Return:
        mat: the matrix filled with cluster labels
    
    """
    
    max_x=max(df['X'])+1
    max_y=max(df['Y'])+1      
    mat = np.empty((max_x,max_y))
    mat.fill(np.nan)            
    
    kml = k_means_cluster(df, a, b, nc)

    
    for l in tqdm(range(len(df))):
                
        xp = df.loc[l,'X'] 
        yp = df.loc[l,'Y']
        mat[int(xp), int(yp)] = kml.iloc[0,l]     
                 
    return mat

def fill_matricies_with_gmmCluster_data(df, a, b, nc):

    """
    Creates an empty matrix and fills it with cluster labels 
    returned by gmm_cluster function
    
    Args:
        df: the DataFrame to be analyzed
        a,b: numerical indices of the column to be clustered
        nc: the number of clusters to be generated
    
    Return:
        mat: the matrix filled with cluster labels
    
    """
    max_x=max(df['X'])+1
    max_y=max(df['Y'])+1      
    mat = np.empty((max_x,max_y))
    mat.fill(np.nan)            
    
    gmml = gmm_cluster(df, a, b, nc)
    
    
    for l in tqdm(range(len(df))):
                
        xp = df.loc[l,'X'] 
        yp = df.loc[l,'Y']
        mat[int(xp), int(yp)] = gmml.iloc[0,l]      
                 
    return mat


def fill_matricies_with_kMeansCluster_AveragedData(df, a, b, nc):

    """
    Creates an empty matrix and fills it with cluster labels returned by
    k_means_cluster function performed on df generated by get_close_points function 
    
    Args:
        df: the DataFrame to be analyzed
        a,b: numerical indices of the column to be clustered
        nc: the number of clusters to be generated
    
    Return:
        mat: the matrix filled with cluster labels on smooth data
        
    """
    max_x=max(df['X'])+1
    max_y=max(df['Y'])+1      
    mat = np.empty((max_x,max_y))
    mat.fill(np.nan)            
    
    cp = []
    
    for j in tqdm(range(len(df))):
        
        xp = df.loc[j,'X'] 
        yp = df.loc[j,'Y']
        parda = get_close_points(df, xp, yp, radius = 2)  
        av_data = parda.iloc[:, a:b].values.mean()
        cp.append(av_data)
    
    cp=pd.DataFrame(cp)
    kma_av = k_means_cluster(cp, 0, 1, nc)  
    kma_av = kma_av.stack()
    
    for l in tqdm(range(len(df))):
                
        xp = df.loc[l, 'X'] 
        yp = df.loc[l, 'Y']
        mat[int(xp), int(yp)] = kma_av.iloc[l:l+1]      
                 
    return mat


def fill_matricies_with_gmmCluster_AveragedData(df, a, b, nc):
    
    """
    Creates an empty matrix and fills it with cluster labels returned by
    gmm_cluster function performed on df generated by get_close_points function 
    
    Args:
        df: the DataFrame to be analyzed
        a,b: numerical indices of the column to be clustered
        nc: the number of clusters to be generated
    
    Return:
        mat: the matrix filled with cluster labels on smooth data
    
    """
    
    max_x=max(df['X'])+1
    max_y=max(df['Y'])+1      
    mat = np.empty((max_x,max_y))
    mat.fill(np.nan)            
    
    cp = []

    for j in tqdm(range(len(df))):
        
        xp = df.loc[j,'X'] 
        yp = df.loc[j,'Y']
        parda = get_close_points(df, xp, yp, radius = 2)
        av_data = parda.iloc[:, a:b].values.mean()
        cp.append(av_data)
        
    cp=pd.DataFrame(cp)    
    gmm_av = gmm_cluster(cp, 0, 1, nc)  
    gmm_av = gmm_av.stack()  
    
    for l in tqdm(range(len(df))):
                
        xp = df.loc[l, 'X'] 
        yp = df.loc[l, 'Y']
        mat[int(xp), int(yp)] = gmm_av.iloc[l:l+1]      
                 
    return mat

def print_images(mat, title):           

    """
    Generates subplot from passed matrix with the possibility to set title as an argument
    
    Args:
        mat: the matrix containing the data to be plotted
        title: the title of each plot
    """
    
    fig = plt.figure(figsize=(5, 10))
    ax = []
    
    ax.append(fig.add_subplot(1, 1, 1))
    
    ax[-1].set_xlabel('X')
    ax[-1].set_ylabel('Y')    
    
    plt.imshow(mat)
        
    ax[0].set_title(title)

