# unsupervised-learning
For this project I will attempt to use Unsupervised to cluster Universities into to two groups, Private and Public.

___
It is **very important to note, I actually have the labels for this data set, but I will NOT use them for the Clustering algorithm, since that is an unsupervised learning algorithm.** 

In this case I will use the labels to try to get an idea of how well the algorithm performed, so the classification report and confusion matrix at the end of this project, don't truly make sense in a real world setting, but they will give me an idea of how well an unsupervised learning algorithm can work.
___

## The Data

I will use a data frame with 777 observations on the following 18 variables.
* Private A factor with levels No and Yes indicating private or public university
* Apps Number of applications received
* Accept Number of applications accepted
* Enroll Number of new students enrolled
* Top10perc Pct. new students from top 10% of H.S. class
* Top25perc Pct. new students from top 25% of H.S. class
* F.Undergrad Number of fulltime undergraduates
* P.Undergrad Number of parttime undergraduates
* Outstate Out-of-state tuition
* Room.Board Room and board costs
* Books Estimated book costs
* Personal Estimated personal spending
* PhD Pct. of faculty with Ph.D.’s
* Terminal Pct. of faculty with terminal degree
* S.F.Ratio Student/faculty ratio
* perc.alumni Pct. alumni who donate
* Expend Instructional expenditure per student
* Grad.Rate Graduation rate
