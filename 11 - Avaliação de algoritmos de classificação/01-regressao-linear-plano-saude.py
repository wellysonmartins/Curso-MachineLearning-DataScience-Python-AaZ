# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 15:36:16 2019

@author: Wellyson
"""

import pandas as pd

base = pd.read_csv('plano-saude.csv')

X = base.iloc[:, 0].values
y = base.iloc[:, 1].values

import numpy as np
correlacao = np.corrcoef(X, y)

X = X.reshape(-1,1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

# b0
regressor.intercept_

# b1
regressor.coef_