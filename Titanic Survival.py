#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix


# In[2]:


df = pd.read_csv('titanic.csv')


# # Data Analysis

# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


plt.figure(figsize=(20,20))
sns.heatmap(df.isnull(), annot=True, cmap='Greens')


# In[9]:


# Age % of null values
(df.Age.isnull().sum()/len(df.Age))*100


# In[10]:


# Cabin % of null values
(df.Cabin.isnull().sum()/len(df.Cabin))*100


# In[11]:


df.Cabin.unique()


# In[12]:


df.head()


# In[13]:


# Drop the Cabin feature
df.drop('Cabin', axis=1, inplace=True)


# In[14]:


df.head()


# In[15]:


df.isnull().sum()


# In[16]:


df.Embarked.unique()
# S - Southampton
# Q - Queenstown
# C - Cherbourg


# In[17]:


# Show the datapoint where the Embarked is null
df[df.Embarked.isnull()]


# In[18]:


# Show all the Embarked value where the Pclass = 1
df.loc[df.Pclass==1,"Embarked"].value_counts()


# In[19]:


# Show all the Embarked value where the Pclass = 2
df.loc[df.Pclass==2,"Embarked"].value_counts()


# In[20]:


# Show all the Embarked value where the Pclass = 3
df.loc[df.Pclass==3,"Embarked"].value_counts()


# In[21]:


# Show all the Embarked value where the Fare = 80.0
df.loc[df.Fare==80.0,"Embarked"].value_counts()


# In[22]:


# Show all the Embarked value where the Ticket = 113572
df.loc[df.Ticket==113572,"Embarked"].value_counts()


# In[23]:


df.Age


# In[24]:


# Mean of Age column
df.Age.mean()


# In[25]:


# Median of Age column
df.Age.median()


# In[26]:


# Mode of Age column
df.Age.mode()


# In[27]:


li = [10,11,12,13,14,15,98]


# In[28]:


# Mean
np.mean(li)
# 12.5 - 24.7


# In[29]:


# Median
np.median(li)
# 12.5 - 13


# In[30]:


# Mode
[24,24,30,60,50,90,56,78]


# In[31]:


# Plot a boxpolot to find out the outliers in the Age columns
# df.boxplot()
plt.figure(figsize=(20,20))
sns.boxenplot(df.Age)
plt.grid()


# In[32]:


df.Age.value_counts()


# In[33]:


df.shape


# In[34]:


# 891 - 30 = 861


# In[35]:


# Fill all the null values in the Age column with its median value
df.Age.fillna(value=df.Age.median(),inplace=True)


# In[36]:


df.isnull().sum()


# In[37]:


# Drop the rows where Embarked is Null
df.dropna(inplace=True)


# In[38]:


df.isnull().sum()


# In[39]:


df.shape


# In[40]:


df.head()


# In[41]:


# Drop off the columns - PassengerId, Name, Ticket
df.drop(['PassengerId', 'Name', 'Ticket'], inplace=True, axis=1)


# In[42]:


df.head()


# In[43]:


# Plot a graph : Strength of Male V/s Strength of Female
df.Sex.value_counts().plot.bar(df.Sex)
plt.grid()


# In[44]:


# Plot a graph : Strength of Survival V/s Strength of Non-survival
df.Survived.value_counts().plot.bar(df.Survived)
plt.grid()


# In[45]:


# Plot a graph to find out the survival & non-survival rate w.r.t. Sex
sns.countplot(x='Survived', data=df, hue='Sex')
plt.grid()


# In[46]:


# Plot a graph to find out the strength of the Pclass
df.Pclass.value_counts().plot.bar(df.Pclass)
plt.grid()


# In[47]:


# Plot a graph to find out the survival & non-survival rate w.r.t. Pclass
sns.countplot(x='Survived', data=df, hue='Pclass')
plt.grid()


# # More EDA can be done - More plots, more graphs, more charts, more rate, percentages, etc.

# In[48]:


df.head()


# In[49]:


# Changing the Age dtype to 'int'
df.Age = df.Age.astype(int)


# In[50]:


df.Fare = round(df.Fare,2)


# # Encoders - To convert the data from the categorical form to numerical form without changing its meaning

# In[51]:


# Label encoding for the column Sex
from sklearn.preprocessing import LabelEncoder


# In[52]:


enc = LabelEncoder()


# In[53]:


df.Sex = enc.fit_transform(df.Sex)


# In[54]:


df.head()


# In[56]:


newdf = df.copy()


# In[57]:


new = df.copy()


# In[60]:


df.head()


# In[61]:


newdf.head()


# In[62]:


pd.get_dummies(newdf['Embarked'])


# In[63]:


df = pd.concat([df, pd.get_dummies(df['Embarked'])], axis=1)


# In[64]:


df.head()


# In[65]:


df.drop(['Embarked', 'C'], axis=1, inplace=True)


# In[66]:


df.head()


# In[67]:


df.info()


# In[68]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), annot=True, cmap='Greens')


# In[69]:


sns.pairplot(df)


# # Feature Importance / Feature Selection

# In[70]:


df.head()


# In[71]:


X = df.iloc[:,1:]
y = df.iloc[:,0]


# In[72]:


X


# In[73]:


y


# In[74]:


from sklearn.ensemble import ExtraTreesClassifier


# In[75]:


feat = ExtraTreesClassifier()


# In[76]:


feat.fit(X,y)


# In[77]:


feat.feature_importances_


# In[78]:


feat_imp = pd.Series(feat.feature_importances_, index=X.columns)
feat_imp.nlargest(8).plot(kind='barh')


# # Spliting the data

# In[79]:


skf = StratifiedKFold(n_splits=5)


# In[80]:


for train_index, test_index in skf.split(X,y):
  X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  y_train, y_test = y.iloc[train_index], y.iloc[test_index]


# In[81]:


y_train


# In[82]:


y_test


# # Model Selection

# In[83]:


classifier = LogisticRegression()


# # Training the model

# In[84]:


classifier.fit(X_train,y_train)


# # Test the model

# In[85]:


y_pred = classifier.predict(X_test)


# # EDA

# In[86]:


final = pd.DataFrame({"Actual":y_test, "Predicted":y_pred})


# In[87]:


final.head()


# In[88]:


sns.heatmap(final.corr(), annot=True, cmap='Greens')


# # Performance Metric - Confusion Matrix

# In[89]:


confusion_matrix(y_test, y_pred)


# In[90]:


accuracy = (98+48)/Total
 = 146/177 = 0.824


# In[91]:


from sklearn.metrics import accuracy_score


# In[92]:


accuracy_score(y_test,y_pred)


# In[93]:


from sklearn.metrics import classification_report


# In[94]:


classification_report(y_test, y_pred)


# In[95]:


import pickle


# In[96]:


pick = pickle.dumps(classifier)


# In[97]:


unpickle = pickle.load(pick)


# In[ ]:




