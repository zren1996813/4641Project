# Predict Viewer Satisfaction of a New Movie
#### Chenyu Dai, Shuge Fan, Haodong Jin, Ziyao Ren
#### Fall 2020 CS 4641 Machine Learning: Group Project
#### Georgia Tech

## Summary Figure

## Introduction/Background
The goal of this project is to predict the viewer satisfaction of a new movie based on its known features, such as its company, director, star, genre, budget, releasing date, runtime, etc. People tend to analyze the movie by predicting their profits. However, we believe that a good movie is not judged by the profit but rather by its public praise. As a result, we approach this dataset from a different perspective, estimating the score it will get from the public audience. The investing companies might not be interested in our project, but the directors and producers who would like to make a movie with a good reputation or even a movie that is considered as a great art piece will definitely be interested in our project. 

## Methods
For the dataset, we found the original dataset from Kaggle (here is the original source). In this dataset, there are over 6,800 data points and 14 features. However, not all data points can be used. During the data preprocessing stage, we have to delete movies that don’t have all 14 features, and those whose budget is 0 because they would affect our model accuracy. We also realized that we cannot utilize all 14 features because some of them might not be related to our result (viewer satisfaction). We’ll also separate those data into two parts, one for training purposes and the other one for testing purposes.

We will apply both supervised learning and unsupervised learning algorithms to achieve the goal of this project. Firstly, we will try to use the correlation matrix and principal component analysis to reduce those irrelevant features. After that, we will conduct the clustering analysis using K-Means and Gaussian Mixture Modelling algorithm to further identify and remove the outliers before training on a sensitive supervised model. For supervised learning, we will choose one algorithm from Decision Tree, Neural Networks, Linear Regression, and Gradient Boosting Tree. We will find the accuracy for each of them based on our current data and pick the one with the highest accuracy as our prediction model. 

## Results
At the end of this project, we want to build a model that can predict the score of a new movie given some of its features with at least 80% accuracy. To test our model, we will use two sets of data. One set is from our testing dataset, which we separated from the original dataset. The other set consists of recent films that are not in our dataset (meaning that they are released recently) to check if our model can predict the score accurately. 

## Discussion

## Reference
