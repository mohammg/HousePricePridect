# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 01:42:51 2021

@author: moham
"""
#%%
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
import pickle
import matplotlib.pyplot as plt
import statsmodels.api as sm
#%%


from sklearn import model_selection

from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import make_scorer, mean_squared_error,mean_absolute_error

df=pd.read_csv('train.csv')
testtrain=pd.read_csv('test.csv')
real=pd.read_csv('sample_submission.csv')
df.drop('Id',axis=1,inplace=True)
df_copy=df.copy()
#%%
def labeldata(data):
    cols=[cname for cname in data.columns if data[cname].dtype==object  ]
    for cname in cols:
        labelencoder_y=LabelEncoder()
        labelencoder_y.fit(data[cname])
        data[cname]= labelencoder_y.transform(data[cname])
        output = open('Price_encoder-{}.pkl'.format(cname), 'wb')
        pickle.dump(labelencoder_y, output)
        output.close()
def loadlabeldata(data):
     cols=[cname for cname in data.columns if data[cname].dtype==object  ]
     for cname in cols:
          pkl_file = open('Price_encoder-{}.pkl'.format(cname), 'rb')
          le_departure = pickle.load(pkl_file) 
          pkl_file.close()
          data[cname] = le_departure.transform(data[cname])
def fix_missing_val_feature(df):
    cat_column=[cname for cname in df.columns if df[cname].dtype!=object  ]
    for cname in df[cat_column]:
        df[cname]=df[cname].fillna(df[cname].mean())
def fix_missing_cat_feature(df):
    cat_column=[cname for cname in df.columns if df[cname].dtype==object  ]
    for cname in df[cat_column]:
        if df[cname].dtype!=object:
            df[cname]=df[cname].astype(object)
        df[cname]=df[cname].fillna('None')
def missing_value(df):
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})
    return missing_value_df[missing_value_df['percent_missing']>0]
def backwardElimination_df(x,y, sl):
    
    regressor_OLS = sm.OLS(y, x).fit()
    coefs = pd.DataFrame({
    'coef': regressor_OLS.params.values,
    'odds ratio': np.exp(regressor_OLS.params.values),
    'pvalue': regressor_OLS.pvalues,
    'name': regressor_OLS.params.index
     }).sort_values(by='pvalue', ascending=False)
    if coefs['pvalue'][0]>0.05:
        x.drop(coefs['name'][0],axis=1,inplace=True)
        print('delete {}'.format(coefs['name'][0]))
        backwardElimination_df(x,y,sl)
    else:
        return regressor_OLS.params.index

#%%
mis_val=missing_value(df)
miss_col= mis_val[mis_val['percent_missing']>5]['column_name']
df_copy.drop(miss_col,axis=1,inplace=True)
testtrain.drop(miss_col,axis=1,inplace=True)
#%%
y=df_copy['SalePrice']
train=df_copy.drop('SalePrice',axis=1)
train_test=df_copy.copy()
all_data=pd.concat([df_copy,testtrain])
fix_missing_cat_feature(all_data)
fix_missing_cat_feature(all_data)
labeldata(all_data)

#%%
fix_missing_cat_feature(train_test)
fix_missing_val_feature(train_test)
fix_missing_cat_feature(testtrain)
fix_missing_val_feature(testtrain)
#%%

#%%
loadlabeldata(train_test)
loadlabeldata(testtrain)

#%%
#selected_column= backwardElimination_df(train,y,0.05)

correlation = train_test.corr()
#%%
c=round(correlation['SalePrice'].sort_values(ascending=False)[1:16], 2)

#%%
test=testtrain[c.index.values]
X=train_test[c.index.values]
#%%
train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)

def get_mae(estimators=500, learning=0.1, depth=3, subsamp=1.0):
    my_model = xgb.XGBRegressor(max_depth=depth, learning_rate=learning, n_estimators=estimators, subsample=subsamp, random_state=0, verbosity=0, objective = 'reg:squarederror')
    my_model.fit(train_X, train_y)
    predictions = my_model.predict(val_X)
    mae = mean_absolute_error(val_y, predictions)
    return mae
import matplotlib.pyplot as plt

min_estimators = 600
min_depth = 4
min_learning = 0.02
min_subsample = 0.95
min_colsample = 10.0

def get_results(plot=True, num_tests=5, type_of_test="estimators"):
    results = {}
    
    for i in range(1, num_tests+1):
        
        if type_of_test == "depth":
           
            results[i] = get_mae(depth=i, estimators=min_estimators, learning=min_learning, subsamp=min_subsample)
        elif type_of_test == "learning":
            results[i*0.01] = get_mae(learning=i*0.01, estimators=min_estimators, depth=min_depth, subsamp=min_subsample)
        elif type_of_test == "estimators":
            results[i*50] = get_mae(estimators=i*50, learning=min_learning, depth=min_depth, subsamp=min_subsample)
        elif type_of_test == "subsample":
            results[i*0.05] = get_mae(subsamp=i*0.05, estimators=min_estimators, learning=min_learning, depth=min_depth)
        else:
            print("type_of_test is not defined correctly")
    
    if plot and results:
        plt.plot(list(results.keys()), list(results.values()))
        plt.xlabel(type_of_test)
        plt.ylabel("MAE")
        plt.show()
    
    return min(results, key=results.get)

min_depth = get_results(num_tests=20, type_of_test="depth")
print("min_depth:", min_depth)
min_learning = get_results(num_tests=20, type_of_test="learning")
print("min_learning:", min_learning)
min_estimators = get_results(num_tests=20)
print("min_estimators:", min_estimators)
min_subsample = get_results(num_tests=20, type_of_test="subsample")
print("min_subsample:", min_subsample)
#%%
my_model_3 = xgb.XGBRegressor(max_depth=min_depth,learning_rate=min_learning,n_estimators=min_estimators,subsample=min_subsample, random_state=0, verbosity=0, objective = 'reg:squarederror')
my_model_3.fit(train_X, train_y)
predictions_3 = my_model_3.predict(val_X)
mae_3 = mean_absolute_error(predictions_3, val_y)
print("Mean Absolute Error:" , mae_3)
#%%
test_preds = my_model_3.predict(test)
output = pd.DataFrame({'Id': testtrain.Id,
                       'SalePrice': test_preds})
output.to_csv('submission-v5.csv', index=False)



