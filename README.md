# How to plot columns of values related to (X,Y) coordinates

In computer terms, images are nothing more than matrices in which a numerical value corresponds to each pair of indices. The goal of this project is to create a software capable of reconstructing, smoothering and analyzing any dataframe containing the information thus codified concerning one or more images.
Reguarding smoothering, the software defines a function capable of collecting, for each single pixel, all the neighbors, where the proximity is defined by the user. Once the coordinates of these nearby points have been collected, the software calculates their average value and replaces it with the value of the initial pixel.
The analyzes conducted are two different types of clustering carried out both on the original values and on those mediated by the close points.

Two essential preliminary steps are:
    - Upload the DataFrame, which through the Pandas library can be carried out in different ways (in the Project.py file a correct and an incorrect one are generated to illustrate the operation).
    - The count of the number of parameters you want to visualize.

```
    #Example of a correct DataFrame
data = pd.DataFrame(np.random.randint(1,100,size=(100, 5)), columns=('X', 'Y','Alfa','Beta', 'Gamma'))

    #Example of an incorrect DataFrame
#data = pd.DataFrame(np.random.randint(-100,0,size=(100, 5)), columns=('X', 'Y','Alfa','Beta', 'Gamma'))

    #Number of parameters
col = (len(data.iloc[0,:]) - 2)
```


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

1. check_correct_coordinates
2. get_close_points
3. k_means_cluster
4. gmm_cluster
5. fill_functions
    - fill_matricies_with_original_data
    - fill_matricies_with_smoother_data
    - fill_matricies_with_kMeansCluster_data
    - fill_matricies_with_gmmCluster_data
 
6. print_functions
    - print_original_images
    - print_smoother_images 
    - print_kMeansCluster_images
    - print_kMeansCluster_AveragedImages
    - print_gmmCluster_images
    - print_gmmCluster_AveragedImage

## 1. check_correct_coordinates(df)
The first function is a control function aimed to verify that the DataFrame is correctly organized, checking whether the columns containing positional data (here called **X** and **Y** respectivelly) have or not positive integer values, which correspond to the positional index on the matrices. 

- df: the only argument it receives is the DataFrame to check.

It returns True in case of correct coordinates and False for incorrect ones.

## 2. get_close_points(df, x, y, radius=2)
The aim of this function is to collect the points within a radius, whose length can be modified in its definition, for each pixel of the image.

- df: is a DataFrame like the one described above.
- x, y: the pixel coordinates from where the function will start to compute the distance.
- radius: radius of my acceptance circumference

It returns the list of points within the radius.

## 3. k_means_cluster(df, nc = 2)
The function to collect the labels for K-Means clustering.

- df: is a DataFrame like the one described above.
- nc: is the number of clusters we want, definible in its definition.

In Project.py it returns a list of labels made by 0 or 1 ordered like the DataFrame which represents membership in one of the two clusters for every single pixel.

## 4. gmm_cluster(df, nc = 2)
The function to collect the labels for GMM clustering.

- df: is a DataFrame like the one described above.
- nc: is the number of clusters we want, definible in its definition.

In Project.py it returns a list of labels made by 0 or 1 ordered like the DataFrame which represents membership in one of the two clusters for every single pixel

## 5. fill_matricies_with_original_data(df)
Here we start to build the images: first of all the function creates a list of empty matricies one for each parameter, reading the size of the images from the maximum value of the coordinate columns; then it scrolls the position data one by one by entering the respective value for each pair of points.

- df: is a DataFrame like the one described above.

It return the list of filled matrcies.

## 6. fill_matricies_with_smoother_data(df)
Similarly to the previous function, it creates the matrices and collects data on the positions, but here for each pair the neighboring points are calculated by calling the function get_close_points(), from whose result the average value is calculated and substituted for the initial pixel.

- df: is a DataFrame like the one described above.

It return the list of filled matrcies.

## 7. fill_matricies_with_kMeansCluster_data(df)
## 8. fill_matricies_with_gmmCluster_data(df)
Those functions call the k_means_cluster() and gmm_cluster() functions to collect the cluster labels for both single value pixels (passing the original DataFrame as an argument) and averaged pixels (passing the DataFrame previously created with the get_close_points() function).

- df: is a DataFrame like the one described above.

They return the lists of filled matrcies.


# Test
To test the efficency of our code there is an additional .py file called Test.py.

This test code stars importing 2 libreries, pandas and numpy that we already know. Then the functions to be tested are imported from the main code.
There are 3 different random generated DataFrame, the first one correct for our code and the remaining two incorrect. The first rows of df1 are printed to show an example of correct DataFrame.
```
df1 = pd.DataFrame(np.random.randint(1,100,size=(100, 5)), columns=('X', 'Y','Alfa','Beta', 'Gamma'))
   
df2 = pd.DataFrame(np.random.randint(-100,0,size=(100, 5)), columns=('X', 'Y','Alfa','Beta', 'Gamma'))
   
df3 = pd.DataFrame(np.random.uniform(size = (100,5)), columns=('X', 'Y','Alfa','Beta', 'Gamma'))

print("\nCorrect DataFrame shape df1: \n\n",df1.head())
```

6 test functions are defined in this file; each of which has the goal to verify that everything works has it should:

## 1. test_check_correct_coordinates(): 
Tests that the check function of main code return True with df1 and False with df2 and df3.

## 2. test_get_close_points():
Tests that with a build-in DataFrame with 3 out of 4 close points the get_close_points() function collect them correctly.

## 3. test_k_means_cluster():
## 4. test_gmm_cluster():
Test that a cluster label for every dataframe point is generated for both methods.

## 5. test_fill():
Uses all the fill functions and verifies that a matrix for every configuration is generated.

If we run Test.py we can see the first 5 columns of the correct DataFrame. To perform the test we need to enter on the Python console the command line:
```
! pytest Test.py
```
that will give us **6 passed** as result.

###### About time
For a 3500 long DataFrame with 2 columns of data the program takes 30 second to print the whole set of subplots.  
