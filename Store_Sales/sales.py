
# coding: utf-8

# In[42]:

import pandas as pd
import numpy as np

train = pd.read_csv("data/train.csv",low_memory=False)
test = pd.read_csv("data/test.csv",low_memory=False)
store = pd.read_csv("data/store.csv",low_memory=False)

train["Order"] = np.arange(train.shape[0])
test["Order"] = np.arange(test.shape[0])
train = pd.merge(train,store,on="Store",how="inner").set_index("Order").ix[np.arange(train.shape[0]), :]
test = pd.merge(test,store,on="Store",how="inner").set_index("Order").ix[np.arange(test.shape[0]), :]


# In[43]:

sales_by_store = train.groupby("Store")
Regression_models = {}

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
#Generating models for each store

def convert_categorical_to_numeric(state_holiday):
    enc = OneHotEncoder()
    state_holiday[state_holiday=='a'] = 1
    state_holiday[state_holiday=='b'] = 2
    state_holiday[state_holiday=='c'] = 3
    enc.fit(state_holiday)
    return enc.transform(state_holiday).toarray()


for store,store_data in sales_by_store:
    X_train = store_data[['Open','Promo','SchoolHoliday']].values
    state_holiday = convert_categorical_to_numeric(store_data[['StateHoliday']].values)
    X_train = np.concatenate((X_train,state_holiday),axis=1)
    Y_train = store_data[['Sales']].values
    clf = LinearRegression()
    clf.fit(X_train,Y_train)
    Regression_models[np.unique(store_data[['Store']].values)[0]] = clf


# In[74]:

import math

result_ids = []
predicted_values = []
#Now to predictions
for tc,test_df in test.iterrows():
    store_id = np.unique(store_data[['Store']].values)[0]
    state_holiday = [0,0,0,0]
    if 'a' == test_df.StateHoliday:
        state_holiday[1] = 1
    elif 'b' == test_df.StateHoliday:
        state_holiday[2] = 1
    elif 'c' == test_df.StateHoliday:
        state_holiday[3] = 1
    else:
        state_holiday[0] = 1 
    
    X_test = np.array([test_df.Open,test_df.Promo,test_df.SchoolHoliday])
    X_test = np.concatenate((X_test,state_holiday),axis=1)
    X_test = np.nan_to_num(X_test)
    regression_model = Regression_models[test_df.Store]
    result_ids.append(test_df.Id)
    predicted_values.append(math.ceil(regression_model.predict(X_test)))
    


# In[75]:

result_df = pd.DataFrame({'Id':result_ids,'Sales':predicted_values})
result_df.to_csv("data/output.csv",index=False)


# In[ ]:



