
# coding: utf-8

# In[16]:

import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta, date
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn.grid_search  import GridSearchCV
from sklearn.svm import SVR

'''
Purpose:- Get first date of given week and year
Inputs:-
	* week -> week number
	* year -> year
Return :- date object
'''

def tofirstdayinisoweek(year, week):
    ret = datetime.strptime('%04d-%02d-1' % (year, week), '%Y-%W-%w')
    if date(year, 1, 4).isoweekday() > 4:
        ret -= timedelta(days=7)
    return ret

'''
Purpose:- To decide whether promo2 is currently active or not
Inputs:-
	dt -> input data
	promo2 -> does store has ever ran promo2
	week -> week number
	year -> year
Return:- True if input date(dt) is in time-frame when promo2 is active
'''

def is_promo2_active(dt,promo2,week,year):
    
    result = []
    for i,promo2 in enumerate(promo2):
        if 0 == promo2:
            result.append(0)
            continue
        
        current_date = datetime.strptime(dt[i][0] +" 00:00:01","%Y-%m-%d %H:%M:%S")
        first_week_day = tofirstdayinisoweek(int(year[i][0]),int(week[i][0]))
        result.append(1 if current_date > first_week_day else 0)
    
    return pd.DataFrame(result)

'''
Purpose:- Convert Categorical features to numeric using OneHotEncoding
Input :- Categorical feature value
Return:- array of list encoded in required format
''' 

def convert_categorical_to_numeric(state_holiday):
    enc = OneHotEncoder()
    state_holiday[state_holiday=='a'] = 1
    state_holiday[state_holiday=='b'] = 2
    state_holiday[state_holiday=='c'] = 3
    enc.fit(state_holiday)
    return enc.transform(state_holiday).toarray()

'''
Pupose:- Find average square residual error
Input:-
	* ypred -> Predicted value by model
	* yactual -> Actual value
Return:- Float value representing error
'''

def compute_error(ypred,yactual):
    return ((sum((ypred-yactual)**2)*1.0)/len(ypred))**0.5

'''
Purpose:- Produce best model for training data
Inputs:-
	* X -> Training set features
	* Y -> Response values
Return:- Best model for the data
'''    
def get_best_model(X,Y,Store):
    
    """
	Approach:-
		- We first divide our data into training and testing test
		- Apply Linear Regression,RandomForest Regressor and Support Vector Regressor on training data
		- compute predictions using each model
		- Compute error for each model
		- return best model (model with low error)	
    """
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.8)
    
    clf_lr = LinearRegression()
    clf_lr.fit(X_train,Y_train.ravel())
    lr_predicted = clf_lr.predict(X_test)
    lr_error = compute_error(lr_predicted,Y_test.ravel())
    
    param_grid = {'n_estimators':[10,20],'max_features':['auto','sqrt'],'min_samples_split':[5]}
    rf_grid = GridSearchCV(estimator=RandomForestRegressor(),param_grid=param_grid,cv=5,n_jobs=1)
    rf_grid.fit(X_train,Y_train.ravel())
    rf_predicted = rf_grid.predict(X_test)
    rf_error = compute_error(rf_predicted,Y_test.ravel())
    
    param_grid = {'kernel':['rbf'],'gamma':[1e-3,1e-4],'C':[1, 10, 100, 1000]}
    svr_grid = GridSearchCV(estimator=SVR(),param_grid=param_grid,cv=5,n_jobs=1)
    svr_grid.fit(X_train,Y_train.ravel())
    svr_predicted = svr_grid.predict(X_test)
    svr_error = compute_error(svr_predicted,Y_test.ravel())

    if lr_error < rf_error and lr_error < svr_error:
		return clf_lr
    elif rf_error < lr_error and rf_error < svr_error:
		return rf_grid
    else:
		return svr_grid

train = pd.read_csv("data/train.csv",low_memory=False)
test = pd.read_csv("data/test.csv",low_memory=False)
store = pd.read_csv("data/store.csv",low_memory=False)

train["Order"] = np.arange(train.shape[0])
test["Order"] = np.arange(test.shape[0])
train = pd.merge(train,store,on="Store",how="inner").set_index("Order").ix[np.arange(train.shape[0]), :]
test = pd.merge(test,store,on="Store",how="inner").set_index("Order").ix[np.arange(test.shape[0]), :]

#Including Competition's effect in our model
train[['CompetitionDistance']] = (train[['CompetitionDistance']] - train[['CompetitionDistance']].min())/(train[['CompetitionDistance']].max() - train[['CompetitionDistance']].min())
test[['CompetitionDistance']] = (test[['CompetitionDistance']] - test[['CompetitionDistance']].mean())/(test[['CompetitionDistance']].max() - test[['CompetitionDistance']].min())
train['CompetitionDistance'] = np.nan_to_num(train[['CompetitionDistance']].values)
test['CompetitionDistance'] = np.nan_to_num(test[['CompetitionDistance']].values)

#Including the existence of Promo2
train[['IsPromo2Active']] = is_promo2_active(train[['Date']].values,train[['Promo2']].values,train[['Promo2SinceWeek']].values,train[['Promo2SinceYear']].values)
test[['IsPromo2Active']] = is_promo2_active(test[['Date']].values,test[['Promo2']].values,test[['Promo2SinceWeek']].values,test[['Promo2SinceYear']].values)


Regression_models = {}
result_ids = []
predicted_values = []
sales_by_store = train.groupby("Store")

feature_list = ['Open','Promo','SchoolHoliday','CompetitionDistance','Promo2']
for store,store_data in sales_by_store:
    X_train = store_data[feature_list].values
    state_holiday = convert_categorical_to_numeric(store_data[['StateHoliday']].values)
    X_train = np.concatenate((X_train,state_holiday),axis=1)
    Y_train = store_data[['Sales']].values
    Store = np.unique(store_data[['Store']].values)[0]
    Regression_models[Store] = get_best_model(X_train,Y_train,Store)

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
    
    X_test = np.array([test_df.Open,test_df.Promo,test_df.SchoolHoliday,test_df.CompetitionDistance,test_df.Promo2])
    X_test = np.concatenate((X_test,state_holiday),axis=1)
    X_test = np.nan_to_num(X_test)
    regression_model = Regression_models[test_df.Store]
    result_ids.append(test_df.Id)
    predicted_values.append(math.ceil(regression_model.predict(X_test)))


result_df = pd.DataFrame({'Id':result_ids,'Sales':predicted_values})
result_df.to_csv("data/output.csv",index=False)



