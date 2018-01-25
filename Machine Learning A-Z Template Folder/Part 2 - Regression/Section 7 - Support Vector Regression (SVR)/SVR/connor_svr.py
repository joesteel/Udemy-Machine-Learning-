

# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
 

#  import data
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[: ,1:2].values
y = dataset.iloc[: ,2:3].values


# scale the dataset
from sklearn.preprocessing import StandardScaler
X_scaler = StandardScaler()
X = X_scaler.fit_transform(X)
y_scaler = StandardScaler()
y = y_scaler.fit_transform(y)


from sklearn.svm import SVR
svm = SVR()
svm.fit(X,y)


y_pred = y_scaler.inverse_transform(svm.predict(X_scaler.transform(6.5)))


# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_scaler.inverse_transform(X), y_scaler.inverse_transform(y), color = 'red')
plt.plot(X_scaler.inverse_transform(X_grid), y_scaler.inverse_transform(svm.predict(X_grid)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()