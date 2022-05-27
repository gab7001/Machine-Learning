#!/usr/bin/env python
# coding: utf-8

# # PCA for iris data set

# ## Import libraries

# In[ ]:


from sklearn import datasets


# In[ ]:


import pandas as pd


# In[ ]:


from sklearn.preprocessing import scale
from sklearn import decomposition


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import numpy as np


# In[ ]:


import plotly.express as px


# In[ ]:


import seaborn as sns


# ## Get the iris dataset

# In[ ]:


iris = datasets.load_iris()


# In[ ]:


print(iris.feature_names)


# In[ ]:


print(iris.target_names)


# In[ ]:


X = iris.data
Y = iris.target


# In[ ]:


X.shape


# In[ ]:


Y.shape


# ## Scale down to 3 components

# In[ ]:


X = scale(X)


# In[ ]:


pca = decomposition.PCA(n_components=3)
pca.fit(X)


# In[ ]:


scores = pca.transform(X)


# In[ ]:


scores_df = pd.DataFrame(scores, columns=['PC1','PC2','PC3'])


# In[ ]:


scores_df


# ## Scatter plot the scores_df

# In[30]:


plt.figure(figsize=(10,8))
plt.scatter(scores[:,0],scores[:,2],c=Y)
plt.xlabel('PC1')
plt.ylabel('PC3')


# ## another viz with labels

# In[29]:


plt.figure(figsize=(10,8))
legend = Y

colors = {0:'#4ab65a',1:'#4a4eb6',2:'#b64a4a'}
labels = {0:'setosa',1:'versicolor',2:'virginica'}

for t in np.unique(legend):
    ix = np.where(legend==t)
    plt.scatter(scores[ix,0],scores[ix,1],c=colors[t],label=labels[t])
    
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


# In[ ]:


legend


# In[ ]:


scores


# ## create df for iris data

# In[ ]:


colname = ['sepal length (cm)',
  'sepal width (cm)',
  'petal length (cm)',
  'petal width (cm)']


# In[ ]:


iris1 = iris.data


# In[ ]:


data1 = pd.DataFrame(data=iris1,columns=colname)


# In[ ]:


scaler1 = StandardScaler()


# In[ ]:


scaler1.fit(data1)


# In[ ]:


scaled_data1 = scaler1.transform(data1)


# In[ ]:


scaled_data1


# In[ ]:


pca1 = decomposition.PCA(n_components = 3)


# In[ ]:


pca1.fit(scaled_data1)


# In[ ]:


scaled_pca1 = pca1.transform(scaled_data1)


# In[ ]:


scaled_pca1


# In[ ]:


colname_scaled_pca1 = ['PC1','PC2','PC3']


# In[ ]:


scaled_pca1_df = pd.DataFrame(data=scaled_pca1, columns=colname_scaled_pca1)


# In[ ]:


scaled_pca1_df


# In[ ]:


px.scatter(scaled_pca1_df,x='PC1',y='PC3')


# In[ ]:


sns.pairplot(scaled_pca1_df)

