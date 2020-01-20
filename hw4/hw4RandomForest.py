# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 22:19:39 2019

@author: cyx01
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import metrics, model_selection, ensemble, preprocessing
#df = pd.DataFrame()
##df = pd.read_json('Movies_and_TV_5.json', orient='index', encoding='utf-8', lines=True)
#df = pd.read_json(open("Movies_and_TV_5.json", "r", encoding="utf8"))
#df.head(3)


data_labeled = pd.DataFrame()
data_labeled = pd.read_excel('movie_data.xlsx', encoding='utf-8')

data_unlabeled = pd.read_excel('movie_data_unlabeled.xlsx')


new_X = []
X = data_labeled.drop('sentiment', axis = 1) # 7086
for value in X['review']:
    new_X.append(value)
y = data_labeled['sentiment']
X_unlabeled = data_unlabeled # 33052
for value in X_unlabeled['review']:
    new_X.append(value)
#yy = data_unlabeled['class']
huge_X = pd.DataFrame(new_X, columns = ['review']) # 40138

vectorizer = CountVectorizer()
new_huge_X = vectorizer.fit_transform(huge_X.review)
tfidfconverter = TfidfTransformer()
new_huge_X1 = tfidfconverter.fit_transform(new_huge_X)
densed_new_huge_X1 = new_huge_X1.todense()
df = pd.DataFrame(densed_new_huge_X1)
X1 = df.iloc[:7999, :]
XX1 = df.iloc[7999:, :]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X1, y, test_size = 0.3333, shuffle = True, stratify = y)
i = 1
for i in range(10):
    model = ensemble.RandomForestClassifier()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_predict)
    print('Accuracy for {} is {}'.format(i, accuracy))
    filename = 'Model {}.pkl'.format(i)
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
        
predictions = []
while len(X1) < 40139:
    model = pickle.load(open('Model 4.pkl', 'rb'))
    i = 0
    for i in range(10):
        yy_predict = model.predict(XX1)
        yy_predictt = yy_predict.tolist()
        predictions.append(yy_predictt)
    predictions_matrix = np.array(predictions)
    daya = pd.DataFrame(predictions_matrix)
    i = 0
    for i in range(len(daya.columns)):
        j = 0
        one_count = 0
        zero_count = 0
        for j in range(10):
            if int(daya.loc[j, i]) == 1: # [row, column]
                one_count += 1
            else:
                zero_count += 1
        if abs(one_count - zero_count) > 7: #confidence = 90%
            if one_count > zero_count:
                label = 1
                X1.loc[len(X1)] = XX1.iloc[i,:]
                y.loc[len(y)] = label
            elif zero_count > one_count:
                label = 0
                X1.loc[len(X1)] = XX1.iloc[i,:]
                print(i)
                y.loc[len(y)] = label
    X_train, X_test, y_train, y_test =   model_selection.train_test_split(X1, y, test_size = 0.3333, shuffle = True, stratify = y)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_predict)
    print('Length of our labeled data is: {}'.format(len(X1)))
    print('Accuracy for the new labeled data is: {}'.format(accuracy))
    with open('Model 4.pkl', 'wb') as file:
        pickle.dump(model, file)
    X1.to_csv('only features - data mininguuuu.csv', index = False)
    y.to_csv('only labiluuuuu - data mininguuuu.csv', index = False)