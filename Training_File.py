
# coding: utf-8

# ### Importing Packages        

# In[14]:


import pandas as pd
import numpy as np


# ### Load Dataset

# In[71]:


dataset = pd.read_csv('data_train.csv')


# In[72]:


dataset


# ### id as index of Dataframe

# In[73]:


dataset.set_index('id',inplace=True)


# In[74]:


dataset


# ### Count the number of NaN values in each column

# In[75]:


for i,j in dataset.isna().sum().iteritems():
    if j != 0:
        print(i,j)


# ### Drop Columns with more NaN

# In[76]:


dataset=dataset.drop(columns=['cat6','cat8'])


# In[77]:


dataset.mean()


# ### Fillna with average values

# In[78]:


dataset=dataset.fillna(dataset.mean())


# In[79]:


dataset.columns


# ### Correlation between columns

# In[80]:


dataset.corr()


# ### Check for dupliate rows

# In[92]:


for i in dataset.duplicated().items():
    if i==True:
        print (i)


# ### Input and Target

# In[98]:


X=dataset.drop('target',axis=1)
Y=dataset.target


# ### Train & Test Split

# In[102]:


from sklearn.cross_validation import train_test_split


# In[104]:


X_train, X_test, Y_train, Y_test =train_test_split(X, Y, test_size = 0.33, random_state = 5)


# ### Fit Model

# In[110]:


from sklearn.linear_model import LogisticRegression


# In[131]:


lr=LogisticRegression()


# In[132]:


lr.fit(X_train,Y_train)


# In[130]:


model=lr.fit(X_train,Y_train)


# ### Save the model to disk

# In[135]:


import pickle
filename = 'finalized_model'
pickle.dump(model, open(filename, 'wb'))


# ### Prediction on Test

# In[114]:


Y_pred=lr.predict(X_test)


# In[115]:


Y_pred


# ### Confusion Matrix

# In[116]:


from sklearn import metrics
cm = metrics.confusion_matrix(Y_test, Y_pred)


# In[117]:


cm


# ### Classification Report

# In[127]:


print(metrics.classification_report(Y_test, Y_pred))


# ### Accuracy Score and Confusion Matrix

# In[126]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);

