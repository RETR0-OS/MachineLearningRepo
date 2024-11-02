# K-Nearest Neighbors (KNN)

# Introduction
K-Nearest Neighbors (KNN) is a simple unsupervised learning algorithm that stores all available cases and classifies new cases based on a similarity measure. KNN has been used in statistical estimation and pattern recognition already in the beginning of 1970’s as a non-parametric technique. Algorithm is based on the feature similarity approach. KNN algorithm assumes that similar things exist in close proximity. In other words, similar things are near to each other. KNN algorithm at the training phase just stores the dataset and when it gets new data, then it classifies that data into a category that is much similar to the available data. KNN algorithm stores all the available data and classifies a new data point based on the similarity. This means when new data appears then it can be easily classified into a well suite category by using KNN algorithm. KNN algorithm can also be used for regression problems. The only difference from the discussed methodology will be using averages of nearest neighbors rather than voting from nearest neighbors.

# Algorithm
- Load the data
- Initialize K to your chosen number of neighbors
- For each example in the data
    - Calculate the distance between the query example and the current example from the data.
    - Add the distance and the index of the example to an ordered collection
- Sort the ordered collection of distances and indices from smallest to largest (in ascending order) by the distances
- Pick the first K entries from the sorted collection
- Get the labels of the selected K entries
- If regression, return the mean of the K labels

# Problems with KNN
- KNN is very slow in classifying the test data.
- KNN is high memory requirement.
- KNN is not good for high dimensional data.
- KNN is not good for categorical features.
- KNN models cannot be saved for later use as it is a lazy learner and hence it does not learn anything in the training phase.
- KNN is sensitive to the scale of the data.


# Mathematical Formulation
- Let's say we have a dataset with two classes, Red and Blue. We have a new data point which we need to classify. We will use KNN algorithm to classify this new data point.
- We will calculate the distance of this new data point from all the other data points in the dataset.
- We will then select the K nearest data points and classify the new data point based on the majority class of these K data points.
- The distance can be calculated using Euclidean distance, Manhattan distance, Minkowski distance, etc.
- The Euclidean distance is the most commonly used distance measure. It is calculated as the square root of the sum of the squared differences between a new point and an existing point.
- The Euclidean distance between two points $(x_1, y_1)$ and $(x_2, y_2)$ is given by:
  $$\sqrt{((x_2 - x_1)^2 + (y_2 - y_1)^2)}$$
- In a more generalized higher dimension $$\sqrt{((x_2 - x_1)^2 + (y_2 - y_1)^2 + (z_2 - z_1)^2) + .....}$$
- 


# Applications
- KNN can be used for both classification and regression predictive problems.
- KNN can be used for hand-writing detection.
- KNN can be used in recommendation systems.