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
For the dataset, we found the original dataset from Kaggle (here is the [original source](https://www.kaggle.com/danielgrijalvas/movies)). In this dataset, there are over 6,800 data points and 14 features. However, not all data points can be used. We have to delete movies that don’t have all 14 features during the data preprocessing stage, and those whose budget is 0 because they would affect our model accuracy. We also realized that we could not utilize all 14 features because some of them might not relate to our result (viewer satisfaction). We’ll also separate those data into two parts: training purposes and the other for testing purposes.

We will apply both supervised learning and unsupervised learning algorithms to achieve the goal of this project. Firstly, we will try to use the correlation matrix and principal component analysis to reduce those irrelevant features. After that, we will conduct the clustering analysis using K-Means and Gaussian Mixture Modelling algorithm to further identify and remove the outliers before training on a sensitive supervised model. We will choose one algorithm from the Decision Tree, Neural Networks, Linear Regression, and Gradient Boosting Tree for supervised learning. We will find the accuracy for each of them based on our current data and pick the one with the highest accuracy as our prediction model.

# Results
At the end of this project, we want to build a model that can predict a new movie's score, given some of its features with at least 80% accuracy. To test our model, we will use two sets of data. One set is from our testing dataset, which we separated from the original dataset. The other set consists of recent films that are not in our dataset (meaning that they are released recently) to check if our model can accurately predict the score. 

# Discussion
Since we want to utilize both supervised learning and unsupervised learning, these two outcomes may be slightly different or colossal different. Thus, we have to choose the optimal one to represent our final answer by constructing a confusion matrix and tuning hyper-parameters to yield the best results. We understand that fewer data points would lead to less accuracy, thus we need to recalculate and rerun the program to get more data to ensure the prediction is more accurate. We believe this model would be able to apply to the market. As we all know, the movie companies' original motivation is to produce movies that can increase the audience's happiness while earning some profits. If they successfully predict the audience's satisfaction with the film in advance, their films will better cater to the viewer's tastes, leading to more profits.

This dataset collects movies from 1986 to 2016. In the future, we want to gather more information about films that are the most recent because the viewer's tastes vary over time. Also, we want to categorize viewers based on their age, gender, or other features to analyze movies that cater to the tastes of different groups of viewers. 

# References
1. Muhammad Hassan Latif, Hammad Afzal, “Prediction of Movies popularity Using Machine Learning Techniques”, IJCSNS International Journal of Computer Science and Network Security, VOL.16 No.8, August 2016 <br />

2. Nahid Quader, Md. Osman Gani, Dipankar Chaki, and Md. Haider Ali, “A Machine Learning Approach to Predict Movie Box-Office Success”, 20th International Conference of Computer and Information Technology (ICCIT), December 2017. <br />

3. Rebecca Eaton, “Unheard Minimalisms: The Functions of the Minimalist Technique in Film Scores”, University of Texas at Austin, May 2008 <br />

4. Sameer Ranjan Jaiswal, Divyansh Sharma, “Predicting Success of Bollywood Movies Using Machine Learning Techniques”, Proceedings of the 10th Annual ACM India Compute Conference, Nov 2017.

