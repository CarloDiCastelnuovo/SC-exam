import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster
from sklearn.mixture import GaussianMixture
from tqdm import tqdm


data = pd.read_excel("c:/Users/carlausss/Desktop/Prova.xlsx")
data = data.dropna()
data = pd.DataFrame(data)

col = 3

def get_close_points(df, x, y, radius = 2):
    x_idx = data.iloc[:, 0]
    y_idx = data.iloc[:, 1]
    dist_sqrd = (x_idx-x)**2 + (y_idx-y)**2
    to_take = np.sqrt(dist_sqrd)<=radius
    
    return data.loc[to_take]

                            #K-MEANS
def km(df, nc = 2 ):
    kml = []
    for i in range(col):
        
        km = cluster.KMeans(n_clusters = nc).fit(data.iloc[:, 2+i:3+i])   
        labels_km = km.labels_ 
        kml.append(labels_km)
    
    return kml


                            #GMM
def gmm(df, nc = 2):
    
    gmml = []
    for i in range(col):
 
        gmm = GaussianMixture(n_components = nc).fit(data.iloc[:, 2+i:3+i])
        labels_gmm = gmm.predict(data.iloc[:, 2+i:3+i])
        gmml.append(labels_gmm)
        
    return gmml


def image(df):

    m = []  
    max_x=max(data.iloc[:,0])+1
    max_y=max(data.iloc[:,1])+1

    for i in range(6*col):
        mat = np.empty((max_x,max_y))
        mat.fill(np.nan)        
        
        m.append(mat)
        
    j = 0

    for j in tqdm(range(len(data))):
        
        xp = data.iloc[j,0] 
        yp = data.iloc[j,1] 
        
        for i in range(col):
        
            m[i][int(xp), int(yp)] = data.iloc[j,2+i]
        
    
        parda = get_close_points(data, xp, yp, radius = 2)
        parda = pd.DataFrame(parda)
        
        for i in range(col):
        
            cp = parda.iloc[:,2+i]
            cp = cp.mean()
    
            m[col+i][int(xp), int(yp)] = cp   
       
    kma = km(data, nc=2)   
    gmma = gmm(data, nc=2)
    
    kma_av = km(parda, nc=2)  
    gmma_av = gmm(parda, nc=2)
    
    kma = pd.DataFrame(kma)
    
    kma_av = pd.DataFrame(kma_av)
    
    gmma = pd.DataFrame(gmma)
    
    gmma_av = pd.DataFrame(gmma_av)
        
    for l in tqdm(range(len(data))):
        xp = data.iloc[l,0] 
        yp = data.iloc[l,1] 
        
        for n in range(col):

            m[2*col+n][int(xp), int(yp)] = kma.iloc[0+n,l]     

            m[3*col+n][int(xp), int(yp)] = gmma.iloc[0+n,l]
        
            m[4*col+n][int(xp), int(yp)] = kma_av.iloc[0+n,l]      

            m[5*col+n][int(xp), int(yp)] = gmma_av.iloc[0+n,l]
        
    return m
   
def print_image(m):           
    
    fig = plt.figure(figsize=(15, 25))
    ax = []

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

    
print_image(image(data))
