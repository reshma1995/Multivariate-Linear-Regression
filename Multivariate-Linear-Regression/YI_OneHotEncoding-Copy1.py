#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# ## Load DataSet from the path

# In[4]:


dataSet = pd.read_json('/home/sampras/MachineLearning/YieldImprovement/stageOneData.json')
dataSet.head()


# ## Separation of Numerical and Categorical Columns

# In[5]:


dataSet_dict = dict(dataSet.dtypes)

numeric_columns=[]
categorial_columns=[]
for i in dataSet_dict:
    if dataSet_dict[i] in ['float64', 'int64', 'float32', 'int32']:
        numeric_columns.append(i)
    else:
        categorial_columns.append(i)


# In[6]:


dataSetCopy = dataSet.copy()


# In[7]:


cat = dataSetCopy[categorial_columns]


# In[8]:


cat.head()


# ## Use OneHot  Encoder to encode the categorical column

# In[9]:


encode = pd.get_dummies(cat)
encode.head()


# ## Concatenate the encoded data to the original data

# In[10]:


dataSetCopyConcat = pd.concat([dataSetCopy,encode], axis=1)
dataSetCopyConcat.head()


# ## Drop the Categorical column from the Concatenated DataFrame

# In[11]:


dataSetCopyDrop = dataSetCopyConcat.drop('Quality',axis=1)
dataSetCopyDrop.head()


# ## Separate independent variable (x) & dependent variable (y)

# In[12]:


xArray = ['R_CHEMICAL_LITRE','R_PRESSURE','R_RICE_IN_KG','R_WATER_IN_LIT','Quality_AVERAGE','Quality_BAD','Quality_GOOD']
x = dataSetCopyDrop[xArray]
y = dataSetCopyDrop.iloc[:,[2,3,4,5,6,8]].values


# ## Split Train(80%) and Test(20%) Data

# In[13]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)


# ## Build the Linear Regression model

# In[14]:


reg=LinearRegression()
reg.fit(x_train,y_train)


# ## Predict value for the 20% test data

# In[15]:


y_pred=reg.predict(x_test)


# ## Check for RMSE and R^2
# - R^2 value ranges from 0 to 1

# In[16]:


from sklearn.metrics import r2_score,mean_squared_error

rmse=np.sqrt(mean_squared_error(y_test,y_pred))
r2=r2_score(y_test,y_pred)
print('RMSE=',rmse)
print('R2 Score',r2)


# # Test model with external test data

# ## Load the test data

# In[17]:


testData = pd.read_json('/home/sampras/MachineLearning/YieldImprovement/test_data.json')
testData


# ## Separate the Numeric and Categorical Columns

# In[18]:


testdataSet_dict = dict(testData.dtypes)

test_numeric_columns=[]
test_categorial_columns=[]
for i in testdataSet_dict:
    if testdataSet_dict[i] in ['float64', 'int64', 'float32', 'int32']:
        test_numeric_columns.append(i)
    else:
        test_categorial_columns.append(i)


# In[19]:


testDataCopy = testData.copy()


# In[20]:


catTest = testDataCopy[test_categorial_columns]
catTest


# ## Use OneHot  Encoder to encode the test categorical column

# In[21]:


encodeTest = pd.get_dummies(catTest)
encodeTest


# ## Concatenate the encoded test data to the original data

# In[22]:


testDataConcat = pd.concat([testDataCopy,encodeTest], axis=1)
testDataConcat


# ## Drop the Categorical column from the Concatenated DataFrame

# In[23]:


testDataDrop = testDataConcat.drop(['Quality'],axis=1)
testDataDrop


# ## Predict value for the test data

# In[24]:


y_pred = reg.predict(testDataDrop)


# In[25]:


y_pred


# ## Convert the Predicted array to dataframe

# In[26]:


df = pd.DataFrame(data=y_pred, columns=["Temp 1", "Temp Manintain time Min", "Temp2","Temp Maintain time","Temp3","Steam inlet temp Deg"])


# In[27]:


df


# In[ ]:




