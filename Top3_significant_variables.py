#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split


# In[2]:


dataset=pd.read_csv("Bike share.csv")
df=dataset.copy()


# # Method 1-correlation coefficient
# 

# In[3]:


corr_matrix=df.corr()
plt.subplot(222)
correlation=corr_matrix["cnt"].sort_values(ascending=True).plot(kind='bar', title='Significance', figsize=(20,9))


# Using correlation coefficient as a metric, the top 3 variables are
# 
# 1.registered
# 2.casual
# 3.temp and atemp

# In[4]:


df.drop(['instant','dteday'],axis=1,inplace=True) 


# In[5]:


x=df.iloc[:,:13]
y=df.iloc[:,13]

x


# # Method 2: f_regression score 

# In[6]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


# In[7]:


def select_features(x_train, y_train, x_test):
 fs = SelectKBest(score_func=f_regression, k=3)
 fs.fit(x_train,y_train)
 x_train_fs = fs.transform(x_train)
 x_test_fs = fs.transform(x_test)
 return x_train_fs, x_test_fs, fs
 


# In[10]:


x_train_fs, x_test_fs, y_train_fs, y_test_fs = train_test_split(x, y, test_size=0.20, random_state=1)
select_features(x_train_fs, y_train_fs, x_test_fs)
for i in range(len(fs.scores_)):
 print('Feature %d: %f' % (i, fs.scores_[i]))


# using  F_regression score metric the top 3 features are
# 
# 1.registered
# 2.casual
# 3.atemp
# 
# 

# # Method 3 mutual_information

# In[11]:


from sklearn.feature_selection import mutual_info_regression


# In[12]:


def select_features(x_train, y_train, x_test):
 fs = SelectKBest(score_func=mutual_info_regression, k='all')
 fs.fit(x_train,y_train)
 x_train_fs = fs.transform(x_train)
 x_test_fs = fs.transform(x_test)
 return x_train_fs, x_test_fs, fs


x_train_fs, x_test_fs, y_train_fs, y_test_fs = train_test_split(x, y, test_size=0.20, random_state=1)

select_features(x_train_fs, y_train_fs, x_test_fs)

for i in range(len(fs.scores_)):
 print('Feature %d: %f' % (i, fs.scores_[i]))


# using Mutual information metric the top 3 features are
# 
# 1.registered 2.casual 3.atemp

# # Based on the results the top 3 significant features are
# 1.Registered
# 2.casual
# 3.atemp

# In[ ]:




