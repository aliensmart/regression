#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 05:06:15 2020

@author: alienmoore
"""

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

"""
#splitting the dataset into the training set and Tes87t set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

"""

"""
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
"""

# Fitting The Decision Tree Regressionto the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, Y)


# Predicting a new result with Polynomial Regression
Y_pred = regressor.predict([[6.5]])

# Vidualising the Decision Tree Regression results
plt.scatter(X, Y, color='red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('truth or bluff(Decision Tree)')
plt.xlabel('position Level')
plt.ylabel('salary')
plt.show()

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('truth or bluff(Decision Tree Regression)')
plt.xlabel('position Level')
plt.ylabel('salary')
plt.show()