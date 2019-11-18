# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 15:17:03 2019

@author: carlausss
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster
from sklearn.mixture import GaussianMixture

    #Import the data from excel (or any other) and get the DataFrame 

data = pd.read_excel("c:/Users/carlausss/Desktop/Prova.xlsx")
data = data.dropna()
data = pd.DataFrame(data)


        #Define the functions to get close points

def get_close_points(df, x, y, radius = 2):
    x_idx = data.iloc[:, 0]
    y_idx = data.iloc[:, 1]
    dist_sqrd = (x_idx-x)**2 + (y_idx-y)**2
    to_take = np.sqrt(dist_sqrd)<=radius
    
    return data.loc[to_take]

nc = 2
                            #K-MEANS

km_alfa = cluster.KMeans(n_clusters = nc).fit(data.iloc[:, 2:3])
km_beta = cluster.KMeans(n_clusters = nc).fit(data.iloc[:, 3:4])

labels_alfa = km_alfa.labels_ 
labels_beta = km_beta.labels_ 

km_dfa = pd.DataFrame(labels_alfa)
km_dfb = pd.DataFrame(labels_beta)

                            #GMM

gmm_alfa = GaussianMixture(n_components=nc).fit(data.iloc[:, 2:3])
gmm_beta = GaussianMixture(n_components=nc).fit(data.iloc[:, 3:4])

lab_alfa = gmm_alfa.predict(data.iloc[:, 2:3])
lab_beta = gmm_beta.predict(data.iloc[:, 3:4])

gmm_dfa = pd.DataFrame(lab_alfa)
gmm_dfb = pd.DataFrame(lab_beta)

 
    #Create empty matrices of the correct lenght to be filled

a = np.empty(((max(data.iloc[:,0])) + 1,(max(data.iloc[:,1])) + 1))
a.fill(np.nan)
b = np.empty(((max(data.iloc[:,0])) + 1,(max(data.iloc[:,1])) + 1))
b.fill(np.nan)
c = np.empty(((max(data.iloc[:,0])) + 1,(max(data.iloc[:,1])) + 1))
c.fill(np.nan)
d = np.empty(((max(data.iloc[:,0])) + 1,(max(data.iloc[:,1])) + 1))
d.fill(np.nan)
e = np.empty(((max(data.iloc[:,0])) + 1,(max(data.iloc[:,1])) + 1))
e.fill(np.nan)
f = np.empty(((max(data.iloc[:,0])) + 1,(max(data.iloc[:,1])) + 1))
f.fill(np.nan)
g = np.empty(((max(data.iloc[:,0])) + 1,(max(data.iloc[:,1])) + 1))
g.fill(np.nan)
h = np.empty(((max(data.iloc[:,0])) + 1,(max(data.iloc[:,1])) + 1))
h.fill(np.nan)


i = 0

while i < len(data):
   
    xp = data.iloc[i,0]             #Positions on the matrix
    yp = data.iloc[i,1] 

    alfa = data.iloc[i,2]            #Values 
    beta = data.iloc[i,3]

    a[int(xp), int(yp)] = alfa          #Fill the frist two matrices 
    b[int(xp), int(yp)] = beta
    
    parda = get_close_points(data, xp, yp, radius = 2)
                                #Get the close points for every (xp,yp) couple 
    al = parda.iloc[:,2]       #and collect their Alfa and Beta values in 2 different
    bet = parda.iloc[:,3]      #pandas DataFrame
    df_al = pd.DataFrame(al)
    df_bet = pd.DataFrame(bet)

    alfa_med = df_al.mean()          #Compute the mean of both the DataFrames
    beta_med= df_bet.mean()
    
    c[int(xp), int(yp)] = alfa_med   #Fill the matrices with averaged values
    d[int(xp), int(yp)] = beta_med
    
    e[int(xp), int(yp)] = km_dfa.iloc[i,0]     #Clustering matrices 
    f[int(xp), int(yp)] = km_dfb.iloc[i,0]

    g[int(xp), int(yp)] = gmm_dfa.iloc[i,0]
    h[int(xp), int(yp)] = gmm_dfb.iloc[i,0]
    
    i = i + 1

              #Print subplots with all the matrices

mat=[a,b,c,d,e,f,g,h]          
fig = plt.figure(figsize=(15,15))
ax = []

for i in range(8):
    
    img = mat[i]
    ax.append( fig.add_subplot(4, 2, i+1) )
    
    ax[-1].set_xlabel('X')
    ax[-1].set_ylabel('Y')    
    
    plt.imshow(img)


ax[0].set_title("Alfa Values")
ax[1].set_title("Beta Values")
ax[2].set_title("Averaged Alfa Values")
ax[3].set_title("Averaged Beta Values")
ax[4].set_title("KM Cluster on Alfa Values")
ax[5].set_title("KM Cluster on Beta Values")
ax[6].set_title("GMM Cluster on Alfa Values")
ax[7].set_title("GMM Cluster on Beta Values")



