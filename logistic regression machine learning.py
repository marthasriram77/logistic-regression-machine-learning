#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# In[3]:


dataframe = pd.read_csv(r"Downloads\ionosphere.data", header=None)


# In[4]:


dataframe


# In[5]:


pd.set_option('display.max_rows', 10000000000)
pd.set_option('display.max_columns', 10000000000)
pd.set_option('display.width', 95)


# In[6]:


print('This DataFrame Has %d Rows and %d Columns'%(dataframe.shape))


# In[7]:


print(dataframe.head())


# In[8]:


features_matrix = dataframe.iloc[:, 0:34]


# In[9]:


target_vector = dataframe.iloc[:, -1]


# In[10]:


print('The Features Matrix Has %d Rows And %d Column(s)'%(features_matrix.shape))
print('The Target Matrix Has %d Rows And %d Column(s)'%(np.array(target_vector).reshape(-1, 1).shape))


# In[11]:


features_matrix_standardized = StandardScaler().fit_transform(features_matrix)


# In[12]:


algorithm = LogisticRegression(penalty='l2', dual=False, tol=1e-4,
C=1.0, fit_intercept=True,
intercept_scaling=1, class_weight=None,
random_state=None, solver='lbfgs',
max_iter=100, multi_class='auto',
verbose=0, warm_start=False, n_jobs=None,
l1_ratio=None)


# In[13]:


Logistic_Regression_Model = algorithm.fit(features_matrix_standardized, target_vector)


# In[14]:


observation = [[1, 0, 0.99539, -0.05889, 0.8524299999999999, 0.02306,
0.8339799999999999, -0.37708, 1.0, 0.0376,
0.8524299999999999, -0.17755, 0.59755, -0.44945, 0.60536,
-0.38223, 0.8435600000000001, -0.38542, 0.58212, -0.32192,
0.56971, -0.29674, 0.36946, -0.47357, 0.56811, -0.51171,
0.41078000000000003, -0.46168000000000003, 0.21266, -0.3409,
0.42267, -0.54487, 0.18641, -0.453]]


# In[15]:


predictions = Logistic_Regression_Model.predict(observation)


# In[16]:


print('The Model Predicted The Observation To Belong To Class %s'%(predictions))


# In[17]:


print('The Algorithm Was Trained To Predict One Of The Two Classes: %s'%(algorithm.classes_))


# In[18]:


print("""The Model Says The Probability Of The Observation We Passed Belonging To Class ['b'] Is %s"""%(algorithm.predict_proba(observation)[0][0]))
print()
print("""The Model Says The Probability Of The Observation We Passed Belonging To Class ['g'] Is %s"""%(algorithm.predict_proba(observation)[0][1]))


# In[ ]:




