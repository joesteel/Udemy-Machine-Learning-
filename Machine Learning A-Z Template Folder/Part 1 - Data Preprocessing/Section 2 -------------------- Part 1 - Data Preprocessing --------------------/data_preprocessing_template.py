# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:57:46 2018

@author: connor
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

"""
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
"""


np.set_printoptions(threshold=np.nan)

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values



# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

