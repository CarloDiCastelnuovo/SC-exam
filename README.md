# CarloDica
## Title

The aim of this project is visualize whatever dataframe in which I have the first two columns filled with position data ('x', 'y') and the rest of them filled with some values i want to visualize. Furthermore the code estimate the close points for every pixel and compute the mean value of it, generating smoother images.

The last step is a clustering phase in which the code estimate for every pixel the belonging to one or another cluster, with two different metodologies: the K-means and the Guassian Mixture Model. This is performed on both original and averaged values, to compare them.
In the end there will be 6 different subplots for every data columns.

In this project there are 2 data columns (called arbitrarily Alfa and Beta), but adding some parameters is possible to work with any number of columns; the code can be improved and generalized by making the user choose the number of columns, and I'm working on it.
