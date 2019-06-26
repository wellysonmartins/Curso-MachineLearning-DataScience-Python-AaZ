# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 11:37:10 2019

@author: Wellyson
"""
import pandas as pd

base = pd.read_csv('risco-credito.csv')
previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder.fit_transform(previsores[:,1])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])

from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores, classe)

# histórica boa, dívida alta, garantias nenhuma, renda > 35
# histórica ruim, dívida alta, ganrantias adequada, renda < 15
resultado = classificador.predict([[0,0,1,2], [3,0,0,0]])

print(classificador.classes_)
print(classificador.class_count_)
print(classificador.class_prior_)




