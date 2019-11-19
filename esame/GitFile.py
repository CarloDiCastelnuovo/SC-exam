import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster
from sklearn.mixture import GaussianMixture

    #Import the data from excel (or any other) and get the DataFrame 

data = pd.read_excel("c:/Users/carlausss/Desktop/Prova.xlsx")
data = data.dropna()
data = pd.DataFrame(data)
            #Create random DataFrame

#a = np.random.randn(100,4)
#data = pd.DataFrame(a, columns=['x','y','Alfa','Beta'])

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

 
    #Create empty matrices of the correct lenght to be filled

def image(df):

    m = []  
    maxx=max(data.iloc[:,0])
    maxy=max(data.iloc[:,1])

    for i in range(8):
        mat = np.empty((int(maxx),int(maxy)))
        mat.fill(np.nan)        
        
        m.append(mat)
        
        i = 0

    while i < len(data):
        
        xp = data.iloc[i,0]             #Positions on the matrix
        yp = data.iloc[i,1] 

        alfa = data.iloc[i,2]            #Values 
        beta = data.iloc[i,3]

        m[0][int(xp), int(yp)] = alfa          #Fill the frist two matrices 
        m[1][int(xp), int(yp)] = beta
    
        parda = get_close_points(data, xp, yp, radius = 2)
                                #Get the close points for every (xp,yp) couple 
        al = parda.iloc[:,2]       #and collect their Alfa and Beta values in 2 different
        bet = parda.iloc[:,3]      #pandas DataFrame
        df_al = pd.DataFrame(al)
        df_bet = pd.DataFrame(bet)

        alfa_med = df_al.mean()          #Compute the mean of both the DataFrames
        beta_med= df_bet.mean()
    
        m[2][int(xp), int(yp)] = alfa_med   #Fill the matrices with averaged values
        m[3][int(xp), int(yp)] = beta_med
       
        kma, kmb = km(data, nc=2)
        gmma, gmmb = gmm(data, nc=2)
    
        kma = pd.DataFrame(kma)
        kmb = pd.DataFrame(kmb)
    
        gmma = pd.DataFrame(gmma)
        gmmb = pd.DataFrame(gmmb)
    
        m[4][int(xp), int(yp)] = kma.iloc[i,0]     #Clustering matrices 
        m[5][int(xp), int(yp)] = kmb.iloc[i,0]

        m[6][int(xp), int(yp)] = gmma.iloc[i,0]
        m[7][int(xp), int(yp)] = gmmb.iloc[i,0]
    
        i = i + 1
        
        return m
   
def print_image(m):           #Print subplots with all the matrices
    
    fig = plt.figure(figsize=(15,15))
    ax = []

    for i in range(8):
    
        ax.append( fig.add_subplot(4, 2, i+1) )
    
        ax[-1].set_xlabel('X')
        ax[-1].set_ylabel('Y')    
    
        images = plt.imshow(m[i])


    ax[0].set_title("Alfa Values")
    ax[1].set_title("Beta Values")
    ax[2].set_title("Averaged Alfa Values")
    ax[3].set_title("Averaged Beta Values")
    ax[4].set_title("KM Cluster on Alfa Values")
    ax[5].set_title("KM Cluster on Beta Values")
    ax[6].set_title("GMM Cluster on Alfa Values")
    ax[7].set_title("GMM Cluster on Beta Values")
    
    return images

print_image(image(data))
