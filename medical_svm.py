#!/usr/bin/env python
# coding: utf-8

# # Medical Diagnosis with Support Vector Machines
# 
# ## Task 1: Import Libraries
# 
# 

# In[1]:


import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ## Task 1: Get Data

# In[2]:


column_names = ["pregnancies", "glucose", "bpressure", "skinfold", "insulin", "bmi", "pedigree", "age", "class"]
df = pd.read_csv('data.csv', names=column_names)
df.shape
df.head()


# ## Task 1: Extract Features

# In[3]:


X = df.iloc[:,:8]


# ## Task 1: Extract Class Labels

# In[4]:


y = df['class']
y.head()


# ## Task 2: Split Dataset

# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
X_train.shape


# ## Task 2: Normalize Features

# In[6]:


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
print(X_train[:5])


# ## Task 3: Training a Support Vector Machine

# In[7]:


clf = svm.SVC(kernel='sigmoid')
clf.fit(X_train, y_train)


# ## Task 3: Decision Boundary

# In[8]:


y_pred = clf.predict(X_train)
print(y_pred)
print(accuracy_score(y_train, y_pred))


# ## Task 3: SVM Kernels

# In[9]:


for k in ('linear', 'poly', 'rbf', 'sigmoid'):
    clf = svm.SVC(kernel=k)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    print(k)
    print(accuracy_score(y_train, y_pred))


# ## Task 4: Instantiating the Best Model

# In[10]:


clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_train)


# ## Task 4: Making a single prediction

# In[11]:


# "pregnancies", "glucose", "bpressure", 
# "skinfold", "insulin", "bmi", 
# "pedigree", "age", "class"
patient = np.array([[ 1, 200, 75, 40, 0, 45, 1.5, 20 ]])
patient = scaler.transform(patient)
clf.predict(patient)


# ## Task 4: Testing Set Prediction

# In[13]:


patient = np.array([ X_test.iloc[1] ])
patient = scaler.transform(patient)
print(clf.predict(patient))


# ## Task 5: Accuracy on Testing Set

# In[14]:


X_test = scaler.transform(X_test)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))


# ## Task 5: Comparison to All-Zero Prediction

# In[15]:


y_zero = np.zeros(y_test.shape)
print(accuracy_score(y_test, y_zero))


# ## Task 5: Precision and Recall

# In[16]:


print(classification_report(y_test, y_pred))


# In[ ]:




