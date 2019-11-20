# How to plot two columns of values related to (x,y) coordinates

The aim of this project is to visualize whatever DataFrame in which I have the first two columns filled with position data ('x', 'y') and the rest of them filled with some values I want to visualize. Furthermore the code estimate the close points for every pixel and compute the mean value of it, generating smoother images.

The last step is a clustering phase in which the code estimate for every pixel the belonging to one or another cluster, with two different metodologies: the K-means and the Guassian Mixture Model (GMM). This is performed on both original and averaged values, to compare them.
In the end there will be 6 different subplots for every data columns.

In this project there are 2 data columns (called arbitrarily Alfa and Beta), but adding some parameters is possible to work with any number of columns; the code can be improved and generalized by making the user choose the number of columns, and I'm working on it.

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
pandas for the DataFrames, numpy for mathematical operations, matplotlib.pyplot to show the subplots, sklearn for the two clusters functions and tqdm if we want to take trace of our cicles


# Functions
The programm is formed by 5 functions:
1. get_close_points
2. km
3. gmm
4. image
5. print_image

## 1. get_close_points(df, x, y, radius=2)
The aim of this function is to collect the points with a radius, whose length can be modified in its definition, for each pixel of the image.

- df: is a DataFrame like the one described above.
- x, y: the pixel coordinates from where the function will start to compute the distance.
- radius: radius of my acceptance circumference

It returns the list of points within the radius.

## 2. km(df, nc = 2)
The function to collect the labels for K-Means clustering.

- df: is a DataFrame like the one described above.
- nc: is the number of clusters I want, definible in its definition.

It returns two lists of 0 or 1 ordered like the DataFrame which represents membership in one of the two clusters.

## 3. gmm(df, nc = 2)
The function to collect the labels for GMM clustering.

- df: is a DataFrame like the one described above.
- nc: is the number of clusters I want, definible in its definition.

It returns two lists of 0 or 1 ordered like the DataFrame which represents membership in one of the two clusters.

## 4. images(df)
This is the main function of the code, it takes only the DataFrame as an argument, but is able to build the 12 matrices I want to visualize. It is possible to better explain its functioning by dividing the operations into 5 steps:
- I: it creates a list of 12 empty matrices, of the lenght of the data and fills them with "nan"; it is important that they are filled        like this and not with zeros as this could be a value of my data and I want to visualize it.
- II: with a for cicle the function takes the coordinates of every pixel and creates the first two images filling the matrices with the       corresponding value.
- III: this step stars calling the get_close_points() function, passing on to it our dataframe and the coordinates of the individual pixels as arguments; then, after it creates a DataFrame with the values of the close points for the starting pixel, it computes the mean of it and fill the matrices with the averaged value.

_The II and III steps are both within a for cicle of the lenght of the data, in order to scroll the list pixel by pixel_

- IV: the function calls the km() and gmm() functions to collect the cluster labels for both single value pixels (passing the original DataFrame as an argument) and averaged pixels (passing the DataFrame previously created with the get_close_points() function), in order to get 4 different lists of labels.
- V: the last step is aimed to create the 4 clustering matrices with the lists on the IV step with another for cicle of the lenght of the data.

The function returns a list of 12 filled matrices.

## 5. print_images(m)
The last function creates a figure made by two columns of subplots sorted as the matrices are created in the previous function.

- m: the list of matrices to be shown.

This function returns nothing and is immediately called by the programm passing image(data) as argument.





















