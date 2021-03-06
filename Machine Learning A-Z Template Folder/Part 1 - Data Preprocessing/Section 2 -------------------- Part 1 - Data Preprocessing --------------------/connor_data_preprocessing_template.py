# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler



np.set_printoptions(threshold=np.nan)

# Importing the dataset
dataset = pd.read_csv('Data.csv')
independant_vars = dataset.iloc[:, :-1].values
dependant_var = dataset.iloc[:, 3].values


# fill empty data with the mean
imputer = Imputer(missing_values = 'NaN', strategy="mean", axis=0, copy=True)
imputer = imputer.fit(independant_vars[:,1:3])
independant_vars[:,1:3] = imputer.transform(independant_vars[:,1:3])


# convert strings based catagories to numerical
le_dv = LabelEncoder()
le_dv.fit(independant_vars[:, 0])
independant_vars[:, 0] = le_dv.transform(independant_vars[:, 0])

le_pur = LabelEncoder()
le_pur.fit(dependant_var)
dependant_var = le_pur.transform(dependant_var)


# the labels that are all equal need to be encded with a hotEncoder to stop them influencing the algo
dv_enc = OneHotEncoder(categorical_features= [0]) 
independant_vars = dv_enc.fit_transform(independant_vars).toarray()







# Splitting the dataset into the Training set and Test set
independant_vars_train, independant_vars_test, dependant_var_train, dependant_var_test = train_test_split(independant_vars, dependant_var, test_size = 0.2, random_state = 0)




# Scale the data
train_scaler = StandardScaler()
train_scaler.fit(independant_vars_train)
independant_vars_train = train_scaler.transform(independant_vars_train)


independant_vars_test = train_scaler.transform(independant_vars_test)



# Feature Scaling
"""
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""