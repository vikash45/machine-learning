# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 09:46:26 2019

@author: Vikash Kumar Singhal
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the datasets
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values
     

#splitting the dataset into learning set and trainings set
"""#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#fitting simple linear regression to the training test
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#predicting the test set result
Y_pred = regressor.predict(X_test)

#visualising the training set result
plt.scatter(X_train,Y_train,color = "red")
plt.plot(X_train,regressor.predict(X_train), color = "blue")
plt.title("salary vs experience(training set)")
plt.xlabel("years of experience")
plt.ylabel("salary")
plt.show()

#visualising the training set result
plt.scatter(X_test,Y_test,color = "blue")
plt.plot(X_train,regressor.predict(X_train), color = "red")
plt.title("salary vs experience(test set)")
plt.xlabel("years of experience")
plt.ylabel("salary")
plt.show()


