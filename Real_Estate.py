#!/usr/bin/env python
# coding: utf-8

# # Real Estate Price Predictor

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv("E:\ML project 1\myenv\data.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing['CHAS'].value_counts()


# In[6]:


housing.describe()


# In[7]:




# In[8]:


import matplotlib.pyplot as plt


# In[9]:


housing.hist(bins=50 , figsize=(20,15))
# bins=50 in Matplotlib's hist function, you're determining how many intervals or bars the histogram should have.
# A higher number of bins can provide more detail about the distribution of the data
# .hist(): This method is used to create histograms for the numerical columns in the DataFrame.
# bins=50: Specifies the number of bins (intervals) for the histogram. In this case, each histogram will be divided into 50 bins.
# figsize=(20,15): Sets the size of the figure (the overall plot) to 20 inches in width and 15 inches in height.


# # Train-Test Splitting

# In[10]:


#for learning prupose
import numpy as np
def split_train_test(data,test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    print(shuffled)
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices] , data.iloc[test_indices]


# **np.random.seed(42)**
# When you set a seed, it initializes the random number generator in such a way that it will produce the same sequence of random numbers every time your code is run.

# **shuffled = np.random.permutation(len(data))**
# len(data): Returns the number of elements in the array or the length of the data.
# np.random.permutation(): This function returns a shuffled copy of the input array (or a range of numbers).
# It shuffles the elements randomly.

# **test_set_size = int(len(data) * test_ratio)**
# This line calculates the size of the testing set based on the specified test_ratio. The test_ratio is a value between 0 and 1,
# representing the proportion of the dataset to be used for testing.

# **test_indices = shuffled[:test_set_size]
# train_indices = shuffled[test_set_size:]**
# 
# In Python, slicing is a way to extract a portion of a sequence (like a list or an array). The general syntax for slicing is start:stop:step, and each of these components is optional:
# 
# start: The index where the slice begins (inclusive).
# stop: The index where the slice ends (exclusive).
# step: The step between indices (default is 1).
# Now, let's delve into the specific slices in your code:
# 
# Testing Set (test_indices):
# 
# shuffled[:test_set_size]: This slice extracts elements from the beginning of the shuffled array up to (but not including) the index test_set_size. It includes elements at indices 0 up to test_set_size - 1.
# For example, if test_set_size is 100, this slice would include elements at indices 0 to 99 from the shuffled array.
# 
# Training Set (train_indices):
# 
# shuffled[test_set_size:]: This slice extracts elements from index test_set_size to the end of the shuffled array. It includes elements at indices test_set_size and onwards.

# In[11]:


# train_set , test_set = split_train_test(housing,0.2)


# In[12]:


# print(f"Rows in train set: {len(train_set)} \nRows in test set: {len(test_set)}\n")


# In[13]:


#sklearn has built in function train_test_split
from sklearn.model_selection import train_test_split
train_set , test_set = train_test_split(housing,test_size= 0.2,random_state = 42)
print(f"Rows in train set: {len(train_set)} \nRows in test set: {len(test_set)}\n")


# In[14]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index , test_index in split.split(housing,housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[15]:


strat_test_set['CHAS'].value_counts()


# In[16]:


strat_train_set['CHAS'].value_counts()


# In[17]:


housing = strat_train_set.copy()


# # Looking for correlation

# In[18]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)
#strong positive correlation if one value is increasing other value will aslo increase
#strong negative correlation if one value is increasing other value will decerease


# In[19]:


from pandas.plotting import scatter_matrix
attributes = ["MEDV","RM","ZN","LSTAT"]
scatter_matrix(housing[attributes],figsize = (12,8))


# There is good negative correlation between LSTAT and MEDV <br>
# when LSTAT will decrease then MEDV will increase. so , LSTAT and MEDV has negative correlation <br>
# and on other side RM and MEDV has good positive correlation <br>
# like you can see RM is highly correlated with MEDV so it is very important attribute for our model that wil help us to predict price 

# In[20]:


housing.plot(kind="scatter",x ='RM' ,y='MEDV', alpha=0.8)
# we plotted Rm on x and MEDV on y


# # Atrribute Combination

# In[21]:


housing['TAXRM'] = housing['TAX']/housing['RM']


# In[22]:


housing.head()


# In[23]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[24]:


housing.plot(kind="scatter",x ='TAXRM' ,y='MEDV', alpha=0.8)
# plotting 


# In[25]:


housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()


# # Missing Attributes

# In[26]:


# To take care of missing values , you have three options 
#     1. get rid of missing data points
#     2. get rid of whole attribute 
#     3. set some values to null values like (0, mean or median)


# In[27]:


# option 1
a = housing.dropna(subset=['RM']) 
a.shape


# In[28]:


#option2
housing.drop("RM",axis=1) 
# note there will be no RM column and also note that the original housing dataframe will remain same


# In[29]:


# option 3
median = housing['RM'].median()
housing['RM'].fillna(median)


# In[30]:


housing.describe() #before we are starting filling missing values


# if we choose option 3 their can also be missing values in test set and also there can can be missing values in RM attribute in features <br>
# we can automate this using sklearn 
# 

# In[31]:


# for doing this we already have a class in sklearn name simpleImputer 
from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(strategy = "median")
imputer.fit(housing)


# In[32]:


imputer.statistics_


# In[33]:


# WE are creating a pipeline where in whichever column missing values are present will be replace with medain
X  = imputer.transform(housing)


# **Fitting the Imputer**:<br> Before using the transform() method, you typically need to fit the imputer to your data using the fit() method. During this fitting process, the imputer calculates the mean, median, or most frequent value of each feature (column) in the dataset.
# 
# **Transforming the Data**:<br> Once the imputer is fitted, you can use the transform() method to replace missing values in your dataset with the values computed during the fitting stage.

# In[34]:


housing_tr = pd.DataFrame(X , columns = housing.columns)


# In[35]:


housing_tr.describe()


# # Scikit Learn Design

# Primarily, three types of objects
# 1. Estimators - It estimates some parameter based on a dataset. Eg. imputer. It has a fit method and transform method. Fit method - Fits the dataset and calculates internal parameters
# 
# 2. Transformers - transform method takes input and returns output based on the learnings from fit(). It also has a convenience function called fit_transform() which fits and then transforms.
# 
# 3. Predictors - LinearRegression model is an example of predictor. fit() and predict() are two common functions. It also gives score() function which will evaluate the predictions.

# ## Feature Scaling

# Primarily, two types of feature scaling methods:
# 1. Min-max scaling (Normalization)<br>
#     (value - min)/(max - min)<br>
#     Sklearn provides a class called **MinMaxScaler** for this <br>
#     
# 2. Standardization <br>
#     (value - mean)/std <br>
#     Sklearn provides a class called **StandardScaler** for this

# # Creating a Pipeline

# In[36]:


#pipeline help us do a work automate and it helps us to do simple series of work
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer' , SimpleImputer(strategy ="median")),
    ('std_scaler',StandardScaler()),
])


# In[37]:


housing_num_tr = my_pipeline.fit_transform(housing)


# In[38]:


housing_num_tr
# its a numpy array


# In[39]:


housing_num_tr.shape


# # Selecting a desired model for Real Estates

# In[40]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
# model = LinearRegression()
# model = DecisionTreeRegressor()
model.fit(housing_num_tr,housing_labels)


# In[41]:


some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]


# In[42]:


prepared_data = my_pipeline.transform(some_data)


# In[43]:


model.predict(prepared_data)


# In[44]:


list(some_labels)


# # Evaluting the model

# In[45]:


from sklearn.metrics import mean_squared_error as MSE
housing_predictions = model.predict(housing_num_tr)
mse = MSE(housing_labels,housing_predictions)
rmse = np.sqrt(mse)


# In[46]:


rmse


# # Using better evalution technique - <br> Cross validation

# In[47]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model , housing_num_tr , housing_labels , scoring ="neg_mean_squared_error", cv = 10)
# cv = 10 means there will be 10 folds
rmse_scores = np.sqrt(-scores)


# In[48]:


rmse_scores


# In[49]:


def print_scores(scores):
    print(f"Scores :{scores}")
    print(f"mean :{scores.mean()}")
    print(f"Standard Deviation :{scores.std()}")


# In[50]:


print_scores(rmse_scores)


# # Model outputs

# 1. Decesion tree:<br>
#     mean :4.417835621175246<br>
#     Standard Deviation :1.0969743910042<br>
# 2. Linear Regression:<br>
#     mean :5.037482786117751<br>
#     Standard Deviation :1.059438240560695<br>
# 3. Random Forrest regressor:<br>
#     mean :3.3573609448849373<br>
#     Standard Deviation :0.6737297361306568

# # Saving the model

# In[51]:


from joblib import dump , load
dump(model , 'Real_estate.joblib')


# # Testing the model omn test data

# In[52]:


X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set['MEDV'].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = MSE(Y_test,final_predictions)
final_rmse = np.sqrt(final_mse)


# In[53]:


final_rmse


# In[54]:


prepared_data[0]


# # Using the model

# In[56]:


from joblib import dump, load
import numpy as np
model = load('Real_Estate.joblib') 
features = np.array([[-5.43942006, 4.12628155, -1.6165014, -0.67288841, -1.42262747,
       -11.44443979304, -49.31238772,  7.61111401, -26.0016879 , -0.5778192 ,
       -0.97491834,  0.41164221, -66.86091034]])
model.predict(features)


# In[ ]:




