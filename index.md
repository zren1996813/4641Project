# Summary Figure
<img src="infographic.png"><br />
# Introduction/Background
The score of movies is a popular method for the audience to understand the overall quality of movies and for critics to determine whether a movie is worth recommending. With the data set found from Kaggle, this project aims to identify features of a movie that strongly correlate with its score and train the model to predict the score based on the features of a given movie. This topic is interesting and useful because we want to help producers to increase the influence of their movies just by improving the important aspects and ignoring the unrelated features. 

With the techniques and knowledge we learned from the class so far, including data processing and unsupervised learning, we dived into the dataset and applied several tactics hoping to find some informative results by doing correlation and PCA analysis. Then, we applied two different clustering algorithms on the transformed dataset to see if movies can be categorized based on the features available. Our goal is to build a prediction algorithm that can output the range of scores that a movie would be rated at to give the producers estimate the success of their movies before releasing.

We did the following steps by midterm:

1. Understand the features through correlation analysis
2. Preprocess the data by reducing the number of columns with PCA
3. Apply K-means and GMM to visualize the categories of movies


# Methods
### Data Preprocessing:

The set of data we are using contains 14 features and 6820 data points. We first remove the data points that have either 0 budget or 0 gross since we believe budget and gross are the two most important features in our machine learning process, and data points with 0 budget or 0 gross could potentially cause inaccuracy. After the first step, we have a set of data with 14 features and 4635 data points. 

Now we notice that there are several categorical features in our dataset, and they are hard to comply with, so we decide to expand categorical features into multiple boolean features. For example, the feature “country” with different values like “U.S, France” is separated into several new boolean features representing if the movie is made in the U.S or if the movie is made in France. With this kind of replacement of features from categorical features to boolean features, we expand our dataset from 14 features to 7772 features.

### Feature Reduction:

#### Step 1.
Since there are 7772 features in our dataset and there are some features that potentially are correlated, we decide to further process our data by doing feature reduction using correlation. We calculate the correlation between the features, and apply a threshold of 0.7. Whenever there are two features that have a correlation higher than 0.7, we delete one of the features from the dataset. After feature reduction, we reduce the number of features from 7772 to 4461.
#### Step 2. PCA
To apply the PCA algorithm, the first thing we want to do is to standardize the data to the same scale because some columns have bigger numbers, like the budget and gross earning columns. Those numbers are in the millions, while other columns are represented by binary numbers like 0 or 1. After data standardization, we run the PCA algorithm and transform the data into fewer dimensions. We chose the first 1985 principal components because they represent a total of about 70% of the cumulative variance explained, and each component has a variance bigger than 1. So we reduced the number of features from 4461 to 1985. The following unsupervised learning algorithms are based on this reduced dataset. Below are the results:<br />
<img src="pca-1.png"><br />
<img src="pca-2.png" width = "650cm"><br />
<img src="pca-3.png" width = "650cm"><br />

### K-Means:
We prefer to predict the score of 4.5 to 10, and each cluster represents a 0.1 score. After I apply the K Means algorithm to cluster our principal components, we evaluate the performance of Kmean from 2 to 100 clusters. As you can see in the elbow figure below, the optimal k means is 6. <br />
<img src="kmeans-1.png" width = "650cm"><br />
In order to see the diagram clearly, we decrease the range to (2,9). Apparently, The optimal cluster should be 6 which is the arm of the diagram.<br />
<img src="kmean-2.png" width = "650cm"><br />
However, when we compute the silhouette score between 3 to 20, the score is always below 0.03 and even turns to negative values. 
<img src="kmean-3.png" width = "900cm" height = "600cm"><br />
Because 6 is the optimal cluster, we visualize the silhouette score between 4 to 8. 
<img src="kmeans4.png" width = "800cm"><br />
The silhouette score is always below 0. Actually, the low silhouette score indicates there are overlapping clusters and even wrong assignments in some cases. To visualize it, the diagram is as follows:
<img src="kmeans-5.png" width = "1000cm"><br />
We apply the optimal cluster 6 into the diagram. Unlucky, there are many clusters overlapped, the center of the clusters is pretty close to each other. In other words, it’s hard to distinguish these clusters since each cluster should keep far away from each other. Thus, based on the evaluation above, k means does not perform well in predicting scores of movies.

### GMM:
At last we applied Gaussian Mixture Models to cluster our principal components. Since the score variable that we are trying to predict lies between 4.5 to 10, we decided to evaluate the performance of GMM from 20 to 55 clusters. We hope that each of the 55 clusters would capture 0.1 range of the score. We determined that the best number of clusters would be 35, which is the joint lowest point of both Akaike information criterion and bayes information criterion.
<img src="gmm-1.png" width = "650cm"><br />
However, when we compute the silhouette scores from 20 to 55 clusters, all of them are less than 0.03, which means that there are overlapping clusters and even wrong assignments in some cases. So we decided GMM does not perform well in predicting scores of movies. And unsupervised learning cannot cluster well due to high dimensionality of our data. 
<img src="gmm-2.png" width = "650cm"><br />

# Results

# Discussion
 

# References
1. Muhammad Hassan Latif, Hammad Afzal, “Prediction of Movies popularity Using Machine Learning Techniques”, IJCSNS International Journal of Computer Science and Network Security, VOL.16 No.8, August 2016 <br />

2. Nahid Quader, Md. Osman Gani, Dipankar Chaki, and Md. Haider Ali, “A Machine Learning Approach to Predict Movie Box-Office Success”, 20th International Conference of Computer and Information Technology (ICCIT), December 2017. <br />

3. Rebecca Eaton, “Unheard Minimalisms: The Functions of the Minimalist Technique in Film Scores”, University of Texas at Austin, May 2008 <br />

4. Sameer Ranjan Jaiswal, Divyansh Sharma, “Predicting Success of Bollywood Movies Using Machine Learning Techniques”, Proceedings of the 10th Annual ACM India Compute Conference, Nov 2017.

