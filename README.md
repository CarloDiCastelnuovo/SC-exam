# How to plot columns of values related to (X,Y) coordinates

In computer terms, images are nothing more than matrices in which a numerical value corresponds to each pair of indices (X,Y). 
The goal of this project is to create a software capable of reconstructing, smoothering and analyzing any dataframe containing the information thus codified concerning one or more images.

Reguarding smoothering, the software defines a function capable of collecting, for each single pixel, all the neighbors, where the proximity is defined by the user. Once the coordinates of these nearby points have been collected, the software calculates their average value and replaces it with the value of the initial pixel.

The analyzes conducted are two different types of clustering carried out both on the original values and on those mediated by the close points. Clustering is a method of unsupervised learning, where each data point is grouped into a cluster, which contains similar kinds of data points.

### K-Means Clustering:

It is an algorithm that classifies samples based on attributes/features into K number of clusters. Clustering or grouping of samples is done by minimizing the distance between the sample and the centroid.The algorithm assigns the centroid and optimize its position based on the distances from the points to it. This is called Hard Assignment: at every single iteration we are certain about the belongs of any data points to particular centroid, and then based on the least-squares distance method, we will optimize the placement of the centroid.

Advantages of K-Means:

1. Running Time

2. Better for high dimensional data.

3. Easy to interpret and Implement.

Disadvantages of K-Means:

1. Assumes the clusters as spherical, so does not work efficiently with complex geometrical shaped data(Mostly Non-Linear)

2. Hard Assignment might lead to wrong grouping.

### Gaussian Mixture Model:

Instead of Hard assigning data points to a cluster, if we are uncertain about where the data points belong or to which group, we use this method. It uses the probability of a sample to determine the feasibility of it belonging to a cluster.

Advantages:

1. Does not assume clusters to be of any geometry. Works well with non-linear geometric distributions as well.

2. Does not bias the cluster sizes to have specific structures as done by K-Means (Circular).

Disadvantages:

1. Uses all the components it has access to, so the initialization of clusters will be difficult when the dimensionality of data is high.

2. Difficult to interpret.


There are three Python files in the repository:
- Functions: the library that contains the functions which, once imported, allow you to create the various images.

- Example: the file showing how to use the library functions.

- Test: the file that tests the efficiency of the library functions.
    
  # Functions.py

## Import
To run the code we need 5 indispensable libraries and an optional one:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import cluster
from sklearn.mixture import GaussianMixture

```
**pandas** for the DataFrames, **numpy** for mathematical operations, **matplotlib.pyplot** to show the subplots, **sklearn** for the two clusters functions and **tqdm** if we want to take trace of our cicles.


## Functions

1. check_correct_coordinates(df)
2. get_close_points(df, x, y, radius = 2)
3. k_means_cluster(df, a, b, nc)
4. gmm_cluster(df, a, b, nc)
5. fill_functions
    - fill_matricies_with_original_data(df, col_name)
    - fill_matricies_with_smoother_data(df, col_name)
    - fill_matricies_with_kMeansCluster_data(df, a, b, nc)
    - fill_matricies_with_gmmCluster_data(df,a, b, nc)
    - fill_matricies_with_kMeansCluster_AveragedData(df, a, b, nc)
    - fill_matricies_with_gmmCluster_AveragedData(df, a, b, nc)
 
6. print_images(m, title)


## 1. check_correct_coordinates(df)
The first function is a control function aimed to verify that the DataFrame is correctly organized, checking whether the columns containing positional data (here called **X** and **Y** respectivelly) have or not positive integer values, which correspond to the positional index on the matrices. 

- df: the only argument it receives is the DataFrame to check.

It returns True in case of correct coordinates and False for incorrect ones.

## 2. get_close_points(df, x, y, radius=2)
The aim of this function is to collect the points within a radius, whose length can be modified in its definition, for each pixel of the image.

- df: is a DataFrame like the one described above.
- x, y: the pixel coordinates from where the function will start to compute the distance.
- radius: radius of my acceptance circumference

It returns a dataframe cointaining the points within the radius.

## 3. k_means_cluster(df, a, b, nc)
The function to collect the labels for K-Means clustering.

- df: is a DataFrame like the one described above.
- a,b: numerical indices of the column to be clustered.
- nc: is the number of clusters we want.

It returns dataframe cointaining the labels ordered like the DataFrame which represents membership in one of the clusters for every single pixel.

## 4. gmm_cluster(df, a, b, nc)
The function to collect the labels for GMM clustering.

- df: is a DataFrame like the one described above.
- a,b: numerical indices of the column to be clustered.
- nc: is the number of clusters desired.

It returns a list of labels ordered like the DataFrame which represents membership in one of the clusters for every single pixel

## 5. fill_matricies_with_original_data(df, col_name)
Here we start to build the images: first of all the function creates a matrix reading the size of the images from the maximum value of the coordinate columns; then it scrolls the position data one by one by entering the respective value for each pair of points.

- df: is a DataFrame like the one described above.
- col_name: is the name of the column to be displyed.

It returns the filled matrix.

## 6. fill_matricies_with_smoother_data(df, col_name)
Similarly to the previous function, it creates the matrix and collects data on the positions, but here for each pair the neighboring points are calculated by calling the function get_close_points(), from whose result the average value is calculated and substituted for the initial pixel.

- df: is a DataFrame like the one described above.
- col_name: is the name of the column to be displyed.

It returns the filled matrix.

## 7. fill_matricies_with_kMeansCluster_data(df, a, b, nc)
## 8. fill_matricies_with_gmmCluster_data(df, a, b, nc)
Similarly to the previous function, they create the matrix and collects data on the positions, but here these functions call the k_means_cluster() and gmm_cluster() functions to collect the cluster labels to be assigned to the respective coordinates.

- df: is a DataFrame like the one described above.
- a,b: numerical indices of the column to be clustered.
- nc: is the number of clusters desired.

They return the matrcies filled with labels.

## 9. fill_matricies_with_kMeansCluster_AveragedData(df, a, b, nc):
## 10. fill_matricies_with_gmmCluster_AveragedData(df, a, b, nc):
Similarly to the previous function, they create the matrix and collects data on the positions, but here these functions firstly call the get_close_points function to generate the dataframe containing the data to be clustered via the apposite functions.

- df: is a DataFrame like the one described above.
- a,b: numerical indices of the column to be clustered.
- nc: is the number of clusters desired.

They return the matrcies filled with labels.

## 11. print_images(mat, title)
Simply takes a matrix and plot it setting axis labels.

-mat: matrix to be plotted.
-title: a string containing the title of the plot.

# Example.py
This file shows how to import the functions and the data to generate the images:

```
from Functions import fill_matricies_with_original_data, fill_matricies_with_smooth_data, fill_matricies_with_kMeansCluster_data, fill_matricies_with_gmmCluster_data
from Functions import fill_matricies_with_kMeansCluster_AveragedData, fill_matricies_with_gmmCluster_AveragedData, check_correct_coordinates, print_images
    
    #Example of a correct DataFrame
data = pd.DataFrame(np.random.randint(1,100,size=(100, 5)), columns=('X', 'Y','Alfa','Beta', 'Gamma'))

    #Example of an incorrect DataFrame
#data = pd.DataFrame(np.random.randint(-100,0,size=(100, 5)), columns=('X', 'Y','Alfa','Beta', 'Gamma'))

    #Example of upload by excel file
#data = pd.read_excel("c:/Users/Desktop/any_file_path/Example.xlsx")

print(data.head())

```
Here is shown how to generate two random dataframe and how to upload one. The beginning of the correct one is printed as a further demonstration.
Then all the fill functions of the library are called with respective arguments, in the example the random generated dataframe is used. Every fill function works independently  on a single column, calling it by the name for the first two function 
```
od_gamma = fill_matricies_with_original_data(data, 'Gamma')
sd_gamma = fill_matricies_with_smooth_data(data, 'Gamma') 
    
print_images(od_gamma, 'Gamma Values') 
print_images(sd_gamma, 'Smoothed Gamma Values')
```
and by the indices for the functions regarding clustering: the indices for the columns in a pandas dataframe are indicated with a couple of integer numbers starting from 0,1 for the first one; so in the case of the 'Alfa' column the indices will be 2, 3:
```
km_mat_alfa = fill_matricies_with_kMeansCluster_data(data, 2, 3, 2) 
gmm_mat_alfa = fill_matricies_with_gmmCluster_data(data, 2, 3, 2)

print_images(km_mat_alfa, 'K-Means Alfa Results')     
print_images(gmm_mat_alfa, 'GMM Alfa Results')



km_av_mat_beta = fill_matricies_with_kMeansCluster_AveragedData(data, 3, 4, 3) 
gmm_av_mat_beta = fill_matricies_with_gmmCluster_AveragedData(data, 3, 4, 3)

print_images(km_av_mat_beta, 'K-Means Smoothed Beta Results')
print_images(gmm_av_mat_beta, 'GMM Smoothed Beta Results') 
```


# Test.py
To test the efficency of our code there is an additional file called Test.py.

This test code stars importing 2 libreries, pandas and numpy that we already know. Then the functions to be tested are imported from the main code.
There are 3 different random generated DataFrame, the first one correct (df1) for our code and the remaining two incorrect (df2, df3). The first rows of df1 are printed to show an example of correct DataFrame.
```
df1 = pd.DataFrame(np.random.randint(1,100,size=(100, 5)), columns=('X', 'Y','Alfa','Beta', 'Gamma'))
   
df2 = pd.DataFrame(np.random.randint(-100,0,size=(100, 5)), columns=('X', 'Y','Alfa','Beta', 'Gamma'))
   
df3 = pd.DataFrame(np.random.uniform(size = (100,5)), columns=('X', 'Y','Alfa','Beta', 'Gamma'))

print("\nCorrect DataFrame shape df1: \n\n",df1.head())
```

6 test functions are defined in this file; each of which has the goal to verify that everything works has it should:

- **1. test_check_correct_coordinates():** 
Tests that the check function of main code return True with df1 and False with df2 and df3.

- **2. test_get_close_points():**
Tests that with a build-in DataFrame the get_close_points() function collects points correctly: running the function with increasing radius will give us a larger number of "close points"

- **3. test_k_means_cluster():**
Test that a cluster label for every dataframe point is generated. Moreover tests that with different number of clusters as a parameter (nc=2, nc=4,...) the numbers that represent the labels increase.

- **4. test_gmm_cluster():**
Test that a cluster label for every dataframe point is generated. Moreover tests that with different number of clusters as a parameter (nc=2, nc=4,...) the numbers that represent the labels increase.

If we run Test.py we can see the first 5 columns of the correct DataFrame. To perform the test we need to enter on the Python console the command line:
```
! pytest Test.py
```
that will give us **4 passed** as result.

###### About time
For a 3500 long DataFrame with 2 columns of data the program takes 30 second to print the whole set of subplots.  
