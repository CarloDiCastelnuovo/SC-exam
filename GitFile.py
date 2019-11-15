# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 15:17:03 2019

@author: carlausss
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_excel("c:/Users/carlausss/Desktop/Prova.xlsx")
data = data.dropna()

data = pd.DataFrame(data)

def get_close_points(df, x, y, radius = 2):
    x_idx = data.iloc[:, 0]
    y_idx = data.iloc[:, 1]
    dist_sqrd = (x_idx-x)**2 + (y_idx-y)**2
    to_take = np.sqrt(dist_sqrd)<=radius
    
    return data.loc[to_take]
 

a = np.empty(((max(data.iloc[:,0])) + 1,(max(data.iloc[:,1])) + 1))
a.fill(np.nan)
b = np.empty(((max(data.iloc[:,0])) + 1,(max(data.iloc[:,1])) + 1))
b.fill(np.nan)
c = np.empty(((max(data.iloc[:,0])) + 1,(max(data.iloc[:,1])) + 1))
c.fill(np.nan)
d = np.empty(((max(data.iloc[:,0])) + 1,(max(data.iloc[:,1])) + 1))
d.fill(np.nan)

cp = []

i = 0

while i < len(data):
   
    xp = data.iloc[i,0]
    yp = data.iloc[i,1] 

    alfa = data.iloc[i,2]
    beta = data.iloc[i,3]

    a[int(xp), int(yp)] = alfa
    b[int(xp), int(yp)] = beta
    
    parda = get_close_points(data, xp, yp, radius = 2)
        
    al = parda.iloc[:,2]
    bet = parda.iloc[:,3]
    df_al = pd.DataFrame(al)
    df_bet = pd.DataFrame(bet)

    alfa_med = df_al.mean()
    beta_med= df_bet.mean()
    
    c[int(xp), int(yp)] = alfa_med
    d[int(xp), int(yp)] = beta_med
    
    i = i + 1

mat=[a,b,c,d]

fig = plt.figure(figsize=(15,15))

ax = []

for i in range(4):
    
    img = mat[i]
    ax.append( fig.add_subplot(3, 2, i+1) )
    
    ax[-1].set_xlabel('X')
    ax[-1].set_ylabel('Y')    
    
    plt.imshow(img)


ax[0].set_title("Alfa Values")
ax[1].set_title("Beta Values")
ax[2].set_title("Averaged Alfa Values")
ax[3].set_title("Averaged Beta Values")