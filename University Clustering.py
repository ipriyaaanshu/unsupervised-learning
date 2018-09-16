
# coding: utf-8

# # Clustering Universities
# 
# For this project I will attempt to use Unsupervised to cluster Universities into to two groups, Private and Public.
# 
# ___
# It is **very important to note, I actually have the labels for this data set, but I will NOT use them for the Clustering algorithm, since that is an unsupervised learning algorithm.** 
# 
# In this case I will use the labels to try to get an idea of how well the algorithm performed, so the classification report and confusion matrix at the end of this project, don't truly make sense in a real world setting, but they will give me an idea of how well an unsupervised learning algorithm can work.
# ___
# 
# ## The Data
# 
# I will use a data frame with 777 observations on the following 18 variables.
# * Private A factor with levels No and Yes indicating private or public university
# * Apps Number of applications received
# * Accept Number of applications accepted
# * Enroll Number of new students enrolled
# * Top10perc Pct. new students from top 10% of H.S. class
# * Top25perc Pct. new students from top 25% of H.S. class
# * F.Undergrad Number of fulltime undergraduates
# * P.Undergrad Number of parttime undergraduates
# * Outstate Out-of-state tuition
# * Room.Board Room and board costs
# * Books Estimated book costs
# * Personal Estimated personal spending
# * PhD Pct. of faculty with Ph.D.’s
# * Terminal Pct. of faculty with terminal degree
# * S.F.Ratio Student/faculty ratio
# * perc.alumni Pct. alumni who donate
# * Expend Instructional expenditure per student
# * Grad.Rate Graduation rate

# ## Import Libraries
# 
# ** Import the libraries usually used for data analysis.**

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Getting the Data

# ** Read in the College_Data file using read_csv.**

# In[2]:


df = pd.read_csv('College_Data',index_col=0)


# **Check the head of the data**

# In[3]:


df.head()


# ** Check the info() and describe() methods on the data.**

# In[4]:


df.info()


# In[5]:


df.describe()


# ## Exploratory Data Analysis
# 
# It's time to create some data visualizations!
# 
# ** Let me create a scatterplot of Grad.Rate versus Room.Board where the points are colored by the Private column. **

# In[6]:


sns.set_style('whitegrid')
sns.lmplot('Room.Board','Grad.Rate',data=df, hue='Private',
           palette='inferno',size=6,aspect=1,fit_reg=False)


# **Creating a scatterplot of F.Undergrad versus Outstate where the points are colored by the Private column.**

# In[7]:


sns.set_style('whitegrid')
sns.lmplot('Outstate','F.Undergrad',data=df, hue='Private',
           palette='inferno',size=6,aspect=1,fit_reg=False)


# ** Creating a stacked histogram showing Out of State Tuition based on the Private column. **

# In[8]:


sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='inferno',size=6,aspect=2)
g = g.map(plt.hist,'Outstate',bins=20,alpha=0.7)


# **Creating a similar histogram for the Grad.Rate column.**

# In[9]:


sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='inferno',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)


# ** I noticed how there seems to be a private school with a graduation rate of higher than 100%.**

# In[10]:


df[df['Grad.Rate'] > 100]


# ** Set that school's graduation rate to 100 so it makes sense.**
# I could have removed this outlier.

# In[11]:


df['Grad.Rate']['Cazenovia College'] = 100


# In[12]:


df[df['Grad.Rate'] > 100]


# In[13]:


sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='inferno',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)


# ## K Means Cluster Creation
# 
# Now it is time to create the Cluster labels!
# 
# ** Importing KMeans from SciKit Learn.**

# In[14]:


from sklearn.cluster import KMeans


# ** Creating an instance of a K Means model with 2 clusters.**

# In[15]:


kmeans = KMeans(n_clusters=2)


# **Fitting the model to all the data except for the Private label.**

# In[16]:


kmeans.fit(df.drop('Private',axis=1))


# ** I should see the cluster center vectors**

# In[18]:


kmeans.cluster_centers_


# ## Evaluation
# 
# There is no perfect way to evaluate clustering if I don't have the labels, however since this is just a project, I do have the labels, so I would take advantage of this to evaluate my clusters.
# 
# ** I create a new column for df called 'Cluster', which is a 1 for a Private school, and a 0 for a public school.**

# In[19]:


def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0


# In[20]:


df['Cluster'] = df['Private'].apply(converter)


# In[21]:


df.head()


# ** Now I create a confusion matrix and classification report to see how well the Kmeans clustering worked without being given any labels.**

# In[22]:


from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(df['Cluster'],kmeans.labels_))
print(classification_report(df['Cluster'],kmeans.labels_))


# Not so bad considering the algorithm is purely using the features to cluster the universities into 2 distinct groups! And yes, there are only 777 records in this data which are very less for an unsupervised learning problem.
# 
# ## End!
