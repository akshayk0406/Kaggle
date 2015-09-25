
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd


# In[12]:

base_dir = "/Users/akshaykulkarni/Documents/Kaggle/Sentiment_Analysis_Movie_Reviews/data/"
train = pd.read_csv(base_dir + "train.tsv",sep="\t")
test = pd.read_csv(base_dir + "test.tsv",sep="\t")

input_cols = ['PhraseId','SentenceId','Phrase']

X_train = train[input_cols]
Y_train = train[['Sentiment']]


# In[13]:

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X_train,Y_train,train_size=0.7)


# In[25]:

#Creating Document Term Matrix from given text. Applying appropriate transformations
from sklearn.feature_extraction.text import CountVectorizer

required_input = np.append(X_train.Phrase,test.Phrase)
vectorizer = CountVectorizer(stop_words='english',lowercase=True,min_df=0.005)
vectorizer.fit(required_input)
dt_matrix = vectorizer.transform(required_input).toarray()


# In[26]:

from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

#Using RandomForest Classifier
rfc = RandomForestClassifier()
param_grid = {'n_estimators': [50],'max_features': ['auto']}

GS_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
GS_rfc.fit(dt_matrix[0:len(x_train),], y_train.values.ravel())
y_test, y_pred = y_test, GS_rfc.predict(dt_matrix[len(x_train):len(x_train)+len(x_test,])


# In[27]:

from sklearn.linear_model import LogisticRegression
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
clf = LogisticRegression()
GS_lr = GridSearchCV(estimator=clf,param_grid=param_grid,cv=5)
GS_lr.fit(dt_matrix[0:len(x_train),], y_train.values.ravel())

y_test, y_pred_lr = y_test, GS_lr.predict(dt_matrix[len(x_train):len(x_train)+len(x_test,])

# In[34]:

accuracy_lr = (y_true['Sentiment']==y_pred_lr).sum()
accuracy_rf = (y_true['Sentiment']==y_pred).sum()

