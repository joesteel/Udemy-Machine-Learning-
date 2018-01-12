# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder



def build_model(X, y):
    print("executing build_model")

    significance = 0.05
    X_opt = X[:, [0,1,2,3,4,5]]
    finished = False
    
    while (finished == False): 
        regressor_OLS = sm.OLS(endog = y, exog = X_opt)
        results = regressor_OLS.fit()
        p_values = results.pvalues
        variable_under_test = np.argmax(p_values)
        print("P_Values", p_values)
        if (p_values.max() >= significance):
            print("removing variable_under_test index", variable_under_test)
            X_opt = np.delete(X_opt, variable_under_test, axis = 1) 
        else: 
            finished = True 
        
    return X_opt



# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


labelEncoder = LabelEncoder()
labelEncoder.fit(X[:, 3])
X[:, 3] = labelEncoder.transform(X[:, 3]) 

hotEncoder = OneHotEncoder(categorical_features = [3])
X = hotEncoder.fit_transform(X).toarray()
X = X[:, 1:]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)


import statsmodels.formula.api as sm
X = np.append(arr = np.ones([50,1]).astype(int), values = X, axis = 1 )
final_X = build_model(X, y)


final_X_train, final_X_test, y_train, y_test = train_test_split(final_X, y, test_size=0.2, random_state=0)
regressor_opt = LinearRegression()
regressor_opt.fit(final_X_train, y_train)

final_y_pred = regressor_opt.predict(final_X_test)



