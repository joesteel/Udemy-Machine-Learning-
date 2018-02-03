# -*- coding: utf-8 -*-


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
 

#  import data
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[: ,1:2].values
y = dataset.iloc[: ,2:3].values
y = y.astype(float)


# -*- coding: utf-8 -*-
from sklearn import tree
clf = tree.DecisionTreeRegressor(random_state = 0)
clf.fit(X,y)
print(clf.predict(6.5))


X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1)) 
plt.scatter(X ,y, color = 'red')
plt.plot(X_grid, clf.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()