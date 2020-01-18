# How to plot columns of values related to (x,y) coordinates

The aim of this project is to visualize whatever DataFrame in which I have the first two columns filled with position data ('x', 'y') and the rest of them filled with some values I want to visualize. The program is able to count the number of data columns that is a fundamental parameter to correctly run the code.
```
data = pd.DataFrame(df, columns=('X', 'Y','Alfa','Beta', 'Gamma'))

col = (len(data.iloc[0,:]) - 2)
```
Furthermore the code estimate the close points for every pixel and compute the mean value of it, generating smoother images.
The last step is a clustering phase in which the code estimate for every pixel the belonging to one or another cluster, with two different metodologies: the K-means and the Guassian Mixture Model (GMM). This is performed on both original and averaged values, to compare them.
In the end there will be 6 different subplots for every data columns.


# Import
To run the code we need 5 indispensable libraries and an optional one:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.mixture import GaussianMixture

from tqdm import tqdm
```
**pandas** for the DataFrames, **numpy** for mathematical operations, **matplotlib.pyplot** to show the subplots, **sklearn** for the two clusters functions and **tqdm** if we want to take trace of our cicles


# Functions
The programm is formed by 6 functions:
1. correct_df
2. get_close_points
3. km
4. gmm
5. images
6. print_images

## 1. correct_df(df)
The first function is a control function aimed to verify that the DataFrame is correctly organized, checking whether the first two columns contain or not positive integer values, which correspond to the positional index on the final matrices.

- df: the only argument it receives is the DataFrame to check.

## 2. get_close_points(df, x, y, radius=2)
The aim of this function is to collect the points within a radius, whose length can be modified in its definition, for each pixel of the image.

- df: is a DataFrame like the one described above.
- x, y: the pixel coordinates from where the function will start to compute the distance.
- radius: radius of my acceptance circumference

It returns the list of points within the radius.

## 3. km(df, nc = 2)
The function to collect the labels for K-Means clustering.

- df: is a DataFrame like the one described above.
- nc: is the number of clusters we want, definible in its definition.

It returns a lists of labels made by 0 or 1 ordered like the DataFrame which represents membership in one of the two clusters for every single pixel

## 4. gmm(df, nc = 2)
The function to collect the labels for GMM clustering.

- df: is a DataFrame like the one described above.
- nc: is the number of clusters we want, definible in its definition.

It returns a lists of labels made by 0 or 1 ordered like the DataFrame which represents membership in one of the two clusters for every single pixel

## 5. images(df)
This is the main function of the code, it takes only the DataFrame as an argument, but is able to build the matrices we want to visualize. It is possible to better explain its functioning by dividing the operations into 5 steps:
- I: it creates a list of empty matrices, of the lenght of the data and fills them with "nan"; it is important that they are filled        like this and not with zeros as this could be a value of our data and we want to visualize it.
- II: with a **for** cicle the function takes the coordinates of every pixel and creates the first images filling the matrices with the corresponding value.
- III: this step stars calling the get_close_points() function, passing to it our DataFrame and the coordinates of the individual pixels as arguments; then, after it creates a DataFrame with the values of the close points for the starting pixel, it computes the mean of it and fill the matrices with the averaged value.

_The II and III steps are both within a **for** cicle of the lenght of the data, in order to scroll the list pixel by pixel_

- IV: the function calls the km() and gmm() functions to collect the cluster labels for both single value pixels (passing the original DataFrame as an argument) and averaged pixels (passing the DataFrame previously created with the get_close_points() function).
- V: the last step is aimed to create the clustering matrices with the lists on the IV step with another **for** cicle of the lenght of the data.

The function returns a list of filled matrices.

## 6. print_images(m)
The last function creates a figure made by columns of subplots sorted as the matrices are created in the previous function.

- m: the list of matrices to be shown.

This function returns the list in which the subplots have been uploaded and is immediately called by the programm passing images(data) as argument.

## Control
At the end of every function's definition we can find a little control performed with an **if** that will raise **ValueError** in the case of malfunction.

# Test
To efficiently run this code the DataFrame must have a specific shape, that is: column 0 and column 1 filled with data position and the rest of the DataFrame filled with any kind of numeric value. 
To test the efficency of our code there is an additional .py file called Test.py.

This test code stars importing 3 libreries, pandas and numpy that we already know, and pytest to test functions. Then the functions to be tested are imported from the main code.
There are 3 different random generated DataFrame, the first one correct for our code and the remaining two incorrect. 
6 test functions, corresponding to the 6 functions of the main code, are defined in this file; each of which has the goal to verify that everything works has it should.

If we run Test.py we can see the first 5 columns of the correct DataFrame. To perform the test we need to enter on the Python console the command line:
```
! pytest Test.py
```
that will give us **6 passed** as result.

###### Note
In this repository we can even find a file called Functions.py in which are just defined the functions described whitout any DataFrame to provide the most general version possbile of the code. 
###### About time
For a 3500 long DataFrame with 2 columns of data the program takes 30 second to print the whole set of subplots.  
