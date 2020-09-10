# How to plot columns of values related to (X,Y) coordinates

In computer terms, images are nothing more than matrices in which a numerical value corresponds to each pair of indices. The goal of this project is to create a software capable of reconstructing, smoothering and analyzing any dataframe containing the information thus codified concerning one or more images.
Reguarding smoothering, the software defines a function capable of collecting, for each single pixel, all the neighbors, where the proximity is defined by the user. Once the coordinates of these nearby points have been collected, the software calculates their average value and replaces it with the value of the initial pixel.
The analyzes conducted are two different types of clustering carried out both on the original values and on those mediated by the close points.

Two essential preliminary steps are the reading of the DataFrame, which through the Pandas library can be carried out in different ways (in the Project.py file a correct and an incorrect one are generated to illustrate the operation), and the count of the number of parameters you want to visualize.

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
The first function is a control function aimed to verify that the DataFrame is correctly organized, checking whether the columns containing positional data have or not positive integer values, which correspond to the positional index on the matrices. 

- df: the only argument it receives is the DataFrame to check.

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

It returns a lists of labels made by 0 or 1 ordered like the DataFrame which represents membership in one of the two clusters for every single pixel

## 4. gmm_cluster(df, nc = 2)
The function to collect the labels for GMM clustering.

- df: is a DataFrame like the one described above.
- nc: is the number of clusters we want, definible in its definition.

It returns a lists of labels made by 0 or 1 ordered like the DataFrame which represents membership in one of the two clusters for every single pixel

## 5. generate_images(df)
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

###### About time
For a 3500 long DataFrame with 2 columns of data the program takes 30 second to print the whole set of subplots.  
