
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV

def get_predictions(clf,X_train,Y_train,X_test,model_name):
	clf.fit(X_train,Y_train)
	predict = clf.predict(X_test)
	print "Testing using " + str(model_name) + " done"
	return predict

def write_predictions(PhraseId,Sentiment,suff,base_dir):
	result_df = pd.DataFrame({'PhraseId':PhraseId,'Sentiment':Sentiment})
	result_df.to_csv(base_dir + 'output_'+str(suff)+'.csv',index=False)

# In[12]:

base_dir = "/Users/akshaykulkarni/Documents/Kaggle/Sentiment_Analysis_Movie_Reviews/data/"
train = pd.read_csv(base_dir + "train.tsv",sep="\t")
test = pd.read_csv(base_dir + "test.tsv",sep="\t")

input_cols = ['PhraseId','SentenceId','Phrase']

X_train = train[input_cols]
Y_train = train[['Sentiment']].values
X_test = test[input_cols]

#Creating Document Term Matrix from given text. Applying appropriate transformations
vectorizer = CountVectorizer(stop_words='english',lowercase=True)
X_train_dt_matrix = vectorizer.fit_transform(X_train.Phrase)
X_test_dt_matrix = vectorizer.transform(X_test.Phrase)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_dt_matrix)
X_test_tfidf = tfidf_transformer.transform(X_test_dt_matrix)

svc_param_grid = {'kernel': ['rbf'],'C': [1, 10, 100, 1000]}
gs_lr = GridSearchCV(SVC(),param_grid=svc_param_grid,cv=5)
predicted = get_predictions(gs_lr,X_train_tfidf,Y_train.ravel(),X_test_tfidf,'Support Vector Machines')
write_predictions(X_test['PhraseId'],predicted,'SVC',base_dir)

'''
clf_sgd = SGDClassifier(loss='hinge', penalty='l2', random_state=42)
predicted = get_predictions(clf_sgd,X_train_tfidf,Y_train.ravel(),X_test_tfidf,'Stochastic Gradient Descent')
write_predictions(X_test['PhraseId'],predicted,'SGD',base_dir)
'''

lr_param_grid = {'penalty':['l1','l2'],'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}
gs_lr = GridSearchCV(LogisticRegression(),param_grid=lr_param_grid,cv=5)
predicted = get_predictions(gs_lr,X_train_tfidf,Y_train.ravel(),X_test_tfidf,'Logistic Regression')
write_predictions(X_test['PhraseId'],predicted,'LR',base_dir)

clf = MultinomialNB()
predicted = get_predictions(clf,X_train_tfidf,Y_train.ravel(),X_test_tfidf,'Multinomial Bayes')
write_predictions(X_test['PhraseId'],predicted,'NB',base_dir)

'''
clf_rf = RandomForestClassifier()
predicted = get_predictions(clf,X_train_tfidf,Y_train.ravel(),X_test_tfidf,'Random Forests')
write_predictions(X_test['PhraseId'],predicted,'RF',base_dir)
'''
