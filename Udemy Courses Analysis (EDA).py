#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install seaborn


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as m
import seaborn as sns


# In[4]:


url = 'Udemy Courses.csv'
df = pd.read_csv(url)


# In[5]:


df.head(10)


# In[6]:


df.tail(10)


# In[7]:


df.info()


# In[9]:


df.shape


# # Cleaning Data

# Unnecessary data

# In[10]:


df.drop(df.loc[:, ['course_title', 'published_timestamp']], axis=1, inplace=True)


# Missing data

# In[11]:


df.isnull().sum()


# Duplicated data

# In[12]:


df.duplicated().sum()


# In[13]:


df = df.drop_duplicates()


# In[14]:


df.duplicated().sum()


# Incorrect data types

# In[16]:


# 1-convert course_id column from int64 to object
df['course_id'] = df['course_id'].astype('object')


# In[17]:


# 2-convert price column from object to int64
for i in df.index:
    if df.loc[i, 'price'] == 'Free':
        df.loc[i, 'price'] = '0'
df['price'] = df['price'].astype('int64')


# Outliers

# In[19]:


plt.style.use('bmh')
plt.figure(figsize=(10, 10))
plt.boxplot(df.loc[:, ['price', 'num_subscribers', 'num_reviews', 'num_lectures']],
            labels=['price', 'num_subscribers', 'num_reviews', 'num_lectures'])
plt.title('Outliers Data')
plt.show()


# In[20]:


# 1-num_subscribers
print(df['num_subscribers'].describe())


# In[21]:


# Cleaning outliers
q1 = df['num_subscribers'].quantile(0.25)
q3 = df['num_subscribers'].quantile(0.75)
iqr = q3 - q1
toprange = q3 + iqr * 1.5
botrange = q1 - iqr * 1.5
for i in df.index:
    if df.loc[i, 'num_subscribers'] > toprange:
        df.loc[i, 'num_subscribers'] = toprange
    if df.loc[i, 'num_subscribers'] < botrange:
        df.loc[i, 'num_subscribers'] = botrange


# In[22]:


# 2-num_reviews
print(df['num_reviews'].describe())


# In[23]:


# Cleaning outliers
q1 = df['num_reviews'].quantile(0.25)
q3 = df['num_reviews'].quantile(0.75)
iqr = q3 - q1
toprange = q3 + iqr * 1.5
botrange = q1 - iqr * 1.5
for i in df.index:
    if df.loc[i, 'num_reviews'] > toprange:
        df.loc[i, 'num_reviews'] = toprange
    if df.loc[i, 'num_reviews'] < botrange:
        df.loc[i, 'num_reviews'] = botrange


# In[24]:


# 3-num_lectures
print(df['num_lectures'].describe())


# In[25]:


# Cleaning data
q1 = df['num_lectures'].quantile(0.25)
q3 = df['num_lectures'].quantile(0.75)
iqr = q3 - q1
toprange = q3 + iqr * 1.5
botrange = q1 - iqr * 1.5
for i in df.index:
    if df.loc[i, 'num_lectures'] > toprange:
        df.loc[i, 'num_lectures'] = toprange
    if df.loc[i, 'num_lectures'] < botrange:
        df.loc[i, 'num_lectures'] = botrange


# In[27]:


plt.figure(figsize=(10, 10))
plt.boxplot(df.loc[:, ['price', 'num_subscribers', 'num_reviews', 'num_lectures']],
            labels=['price', 'num_subscribers', 'num_reviews', 'num_lectures'])
plt.title('Cleaned Data')
plt.show()


# 
# # Exploring Data

# In[28]:


df.head(10)


# In[29]:


df.tail(10)


# In[30]:


df.info()


# In[31]:


df.describe()


# In[33]:


# Paid Courses
df['is_paid'].value_counts()


# In[49]:


plt.figure(figsize=(7, 7))
plt.pie(df['is_paid'].value_counts(), explode=[0, 0.3], autopct='%1.1f%%', shadow=True)
plt.legend(['Paid', 'Not paid'], loc='upper right')
plt.title('Paid Courses', loc='left')
plt.show()


# In[36]:


# Prices
df['price'].describe()


# In[39]:


# The most 20 frequent prices of courses
df['price'].value_counts().head(20)


# In[40]:


price = np.array(df['price'])
plt.figure(figsize=(15, 10))
plt.xlim(0, 210)
plt.xticks(np.arange(0, 201, 20))
plt.ylim(0, 850)
plt.yticks(np.arange(0, 851, 50))
plt.hist(price, color='r', rwidth=0.5, align='left', bins=[i for i in range(202)])
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Price frequency distribution')
plt.show()


# In[41]:


# Free courses
print(df[df['price'] == 0]['price'].count())


# In[43]:


# Correlations between price, num_subscribers, num_reviews, and 'num_lectures'
plt.figure(figsize=(10, 10))
sns.heatmap(df.loc[:, ['price', 'num_subscribers', 'num_reviews', 'num_lectures']].corr(), annot=True)
plt.title('Correlations')
plt.show()


# In[44]:


# level column
df['level'].describe()


# In[45]:


# Levels of courses
df['level'].value_counts()


# In[48]:


plt.figure(figsize=(7, 7))
plt.pie(df['level'].value_counts(), explode=[0.1, 0.1, 0.1, 0.1], autopct='%1.1f%%', shadow=True)
plt.legend(['All Levels', 'Beginners', 'Intermediates', 'Experts'], loc='upper right')
plt.title('Levels of courses')
plt.show()


# In[50]:


# Average prices in levels of courses
df.groupby('level')['price'].mean()


# In[51]:


plt.figure(figsize=(7, 7))
plt.pie(np.array(df.groupby('level')['price'].mean()), explode=[0.07, 0.07, 0.07, 0.07], autopct='%1.1f%%', shadow=True)
plt.legend(['All Levels', 'Beginners', 'Intermediates', 'Experts'], loc='upper right')
plt.title('Average price in levels of courses')
plt.show()


# In[52]:


# Average number of subscribers in levels of courses
df.groupby('level')['num_subscribers'].mean()


# In[53]:


plt.figure(figsize=(7, 7))
plt.pie(np.array(df.groupby('level')['num_subscribers'].mean()), explode=[0.07, 0.07, 0.07, 0.07], autopct='%1.1f%%', shadow=True)
plt.legend(['All Levels', 'Beginners', 'Intermediates', 'Experts'], loc='upper right')
plt.title('Average number of subscribers in levels of courses')
plt.show()


# In[54]:


# Average number of lectures in levels of courses
df.groupby('level')['num_lectures'].mean()


# In[55]:


plt.figure(figsize=(7, 7)) 
plt.pie(np.array(df.groupby('level')['num_lectures'].mean()), explode=[0.07, 0.07, 0.07, 0.07], autopct='%1.1f%%', shadow=True)
plt.legend(['All Levels', 'Beginners', 'Intermediates', 'Experts'], loc='upper right')
plt.title('Average number of lectures in levels of courses')
plt.show()


# In[56]:


# Duration
df['content_duration'].describe()


# In[57]:


# The most 10 frequent durations
df['content_duration'].value_counts().head(10)


# In[58]:


# Subjects
df['subject'].describe()


# In[59]:


# Subjects of courses
df['subject'].value_counts()


# In[60]:


plt.figure(figsize=(7, 7))
plt.pie(df['subject'].value_counts(), explode=[0.1, 0.1, 0.1, 0.1], autopct='%1.1f%%', shadow=True)
plt.legend(['Web Development', 'Business Finance', 'Musical Instruments', 'Graphic Design'], loc='upper right')
plt.title('Subjects of courses')
plt.show()


# In[61]:


# Levels of subjects of courses
df.groupby('subject')['level'].value_counts()


# In[63]:


plt.figure(figsize=(12, 12))
LevSubjs = df.groupby('subject')['level'].value_counts()
labels = ['All levels BF', 'Beginners BF', 'Intermediate BF', 'Expert BF',
          'All levels GD', 'Beginners GD', 'Intermediate GD', 'Expert GD',
          'All levels MI', 'Beginners MI', 'Intermediate MI', 'Expert MI',
          'All levels WD', 'Beginners WD', 'Intermediate WD', 'Expert WD']
explodes = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
            0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
plt.pie(LevSubjs, autopct='%1.1f%%', labels=labels, explode=explodes)
plt.show()


# In[65]:


# Levels of top courses' prices
topCourses = df[df['price'] == df['price'].max()]
topCourLev = topCourses['subject'].value_counts()
topCourLev


# In[68]:


plt.figure(figsize=(7, 7))
plt.pie(topCourLev, autopct='%1.1f%%', explode=[0.1, 0.1, 0.1, 0.1], shadow=True)
plt.title('Top Courses Levels')
plt.legend(['Business Finance', 'Web Development', 'Musical Instruments', 'Graphic Design'], loc='upper right')
plt.show()


# In[ ]:




