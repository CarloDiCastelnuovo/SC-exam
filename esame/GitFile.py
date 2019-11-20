import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

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

                            #K-MEANS
def km(df, nc = 2 ):
    km_alfa = cluster.KMeans(n_clusters = nc).fit(data.iloc[:, 2:3])
    km_beta = cluster.KMeans(n_clusters = nc).fit(data.iloc[:, 3:4])

    labels_alfa = km_alfa.labels_ 
    labels_beta = km_beta.labels_ 

    return labels_alfa, labels_beta


                            #GMM
def gmm(df, nc = 2):
    gmm_alfa = GaussianMixture(n_components = nc).fit(data.iloc[:, 2:3])
    gmm_beta = GaussianMixture(n_components = nc).fit(data.iloc[:, 3:4])

    lab_alfa = gmm_alfa.predict(data.iloc[:, 2:3])
    lab_beta = gmm_beta.predict(data.iloc[:, 3:4])

    return lab_alfa, lab_beta


def images(df):

    m = []  
    max_x=max(data.iloc[:,0])+1
    max_y=max(data.iloc[:,1])+1

    for i in range(12):    #Create empty matrices of the correct lenght to be filled

        mat = np.empty((max_x,max_y))
        mat.fill(np.nan)        
        
        m.append(mat)
        
    j = 0

    for j in tqdm(range(len(data))):
        
        xp = data.iloc[j,0]             #Positions on the matrix
        yp = data.iloc[j,1] 

        m[0][int(xp), int(yp)] = data.iloc[j,2]           #Fill the frist two matrices 
        m[1][int(xp), int(yp)] = data.iloc[j,3] 
    
        parda = get_close_points(data, xp, yp, radius = 2)
                                    #Get the close points for every (xp,yp) couple 
        parda = pd.DataFrame(parda)
        
        al = parda.iloc[:,2]        #and collect their Alfa and Beta values in 2 different
        bet = parda.iloc[:,3]       #pandas DataFrame

        al = al.mean()          #Compute the mean of both the DataFrames
        bet = bet.mean()
    
        m[2][int(xp), int(yp)] = al   #Fill the matrices with averaged values
        m[3][int(xp), int(yp)] = bet
       
    kma, kmb = km(data, nc=2)               #Call the cluster functions to get the labels
    gmma, gmmb = gmm(data, nc=2)
    
    kma_av, kmb_av = km(parda, nc=2)  
    gmma_av, gmmb_av = gmm(parda, nc=2)
    
    kma = pd.DataFrame(kma)
    kmb = pd.DataFrame(kmb)
    
    kma_av = pd.DataFrame(kma_av)
    kmb_av = pd.DataFrame(kmb_av)
    
    gmma = pd.DataFrame(gmma)
    gmmb = pd.DataFrame(gmmb)
    
    gmma_av = pd.DataFrame(gmma_av)
    gmmb_av = pd.DataFrame(gmmb_av)
    
    l=0
    
    for l in tqdm(range(len(data))):
        xp = data.iloc[l,0] 
        yp = data.iloc[l,1] 

        m[4][int(xp), int(yp)] = kma.iloc[l,0]     #Clustering matrices 
        m[5][int(xp), int(yp)] = kmb.iloc[l,0]

        m[6][int(xp), int(yp)] = gmma.iloc[l,0]
        m[7][int(xp), int(yp)] = gmmb.iloc[l,0]
        
        m[8][int(xp), int(yp)] = kma_av.iloc[l,0]     
        m[9][int(xp), int(yp)] = kmb_av.iloc[l,0]

        m[10][int(xp), int(yp)] = gmma_av.iloc[l,0]
        m[11][int(xp), int(yp)] = gmmb_av.iloc[l,0]
        
    return m
   
def print_images(m):           #Print subplots with all the matrices
    
    fig = plt.figure(figsize=(15, 25))
    ax = []

    for i in range(12):
    
        ax.append( fig.add_subplot(8, 2, i+1) )
    
        ax[-1].set_xlabel('X')
        ax[-1].set_ylabel('Y')    
    
        plt.imshow(m[i])


    ax[0].set_title("Alfa Values")
    ax[1].set_title("Beta Values")
    ax[2].set_title("Averaged Alfa Values")
    ax[3].set_title("Averaged Beta Values")
    ax[4].set_title("KM Cluster on Alfa Values")
    ax[5].set_title("KM Cluster on Beta Values")
    ax[6].set_title("GMM Cluster on Alfa Values")
    ax[7].set_title("GMM Cluster on Beta Values")
    ax[8].set_title("KM Cluster on Averaged Alfa Values")
    ax[9].set_title("KM Cluster on Averaged Beta Values")
    ax[10].set_title("GMM Cluster on Averaged Alfa Values")
    ax[11].set_title("GMM Cluster on Averaged Beta Values")
    
print_image(image(data))
