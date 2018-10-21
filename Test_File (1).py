
# coding: utf-8

# ### Importing Libraries

# In[4]:


import pandas as pd
import numpy as np


# ### Load Test Data

# In[22]:


dataset = pd.read_csv('data_test.csv')


# In[24]:


dataset


# In[25]:


dataset.set_index('id',inplace=True)


# ### Count the number of NaN values in each column

# In[26]:


for i,j in dataset.isna().sum().iteritems():
    if j != 0:
        print(i,j)


# ### Drop Columns with more NaN

# In[27]:


dataset=dataset.drop(columns=['cat6','cat8'])


# ### Fillna with average values

# In[28]:


dataset=dataset.fillna(dataset.mean())


# ### Correlation

# In[9]:


dataset.corr()


# ### Duplicates

# In[10]:


for i in dataset.duplicated().items():
    if i==True:
        print (i)


# ### Load Model

# In[20]:


import pickle
loaded_model = pickle.load(open("finalized_model", 'rb'))


# ### Prediction on Unseen data using Model

# In[29]:


Predicted=loaded_model.predict(dataset)


# In[32]:


Predicted.shape


# In[33]:


pd.DataFrame(Predicted)


# In[34]:


dataset.index.values


# ### Creating DataFrame with id and target values

# In[38]:


dataset2=pd.DataFrame(Predicted)


# In[43]:


dataset2


# In[41]:


dataset2['id']=dataset.index.values


# In[42]:


dataset2


# In[56]:


dataset2.columns


# ### Renaming Column names

# In[51]:


dataset2 = dataset2.rename(columns={0: 'Target'})


# In[54]:


dataset2.set_index('id',inplace=True)


# In[55]:


dataset2


# ### Looking for Target=1

# In[61]:


Predicted.nonzero()


# ### Calculating corresponding probabilities

# In[64]:


loaded_model.predict_proba(dataset)


# ### Probability of Target being 1

# In[77]:


loaded_model.predict_proba(dataset)[:,1]


# ### Dataset with Probabilities

# In[66]:


dataset3=dataset2.drop('Target',axis=1)


# In[67]:


dataset3


# In[78]:


dataset3['target']=loaded_model.predict_proba(dataset)[:,1]


# In[79]:


dataset3


# ### Creating a csv to write the result

# In[83]:


dataset3.to_csv("Output.csv")


# ### Reading output file

# In[86]:


pd.read_csv("Output.csv")

