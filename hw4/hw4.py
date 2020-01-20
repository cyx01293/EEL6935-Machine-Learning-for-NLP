# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 15:36:27 2019

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


X_train = data_labeled.loc[:6400, 'review'].values
y_train = data_labeled.loc[:6400, 'sentiment'].values
X_test = data_labeled.loc[6401:, 'review'].values
y_test = data_labeled.loc[6401:, 'sentiment'].values
XX1 = data_unlabeled.loc[:,'review'].values
#X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.3333, shuffle = True, stratify = y)
#X_train=X_train.values
#y_train=y_train.values
#X_test=X_test.values
#y_test=y_test.values
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


tokenizer_obj = Tokenizer()
total_reviews = np.concatenate((X_train, X_test), axis=0)
#total_reviews = vectorizer.fit_transform(total_reviews.ravel())
tokenizer_obj.fit_on_texts(total_reviews) 

# pad sequences
max_length = max([len(s.split()) for s in total_reviews])
#max_length=500

# define vocabulary size
vocab_size = len(tokenizer_obj.word_index) + 1

X_train_tokens =  tokenizer_obj.texts_to_sequences(X_train)
X_test_tokens = tokenizer_obj.texts_to_sequences(X_test)
X_unlabeled_tokens = tokenizer_obj.texts_to_sequences(XX1)

X_train_pad = pad_sequences(X_train_tokens, maxlen=max_length, padding='pre')
X_test_pad = pad_sequences(X_test_tokens, maxlen=max_length, padding='pre')
X_unlabeled_pad = pad_sequences(X_unlabeled_tokens, maxlen=max_length, padding='pre')


from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU, Dropout
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint

EMBEDDING_DIM = 10

print('Build model...')

model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=max_length))
model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
#model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
#model.add(LSTM(32, return_sequences=True))
#model.add(LSTM(128))
#model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
#callbacks = [EarlyStopping(monitor='val_loss', patience=2),
#             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Summary of the built model...')
print(model.summary())

print('Train...')

#history = model.fit(X_train_pad, y_train, batch_size=128, epochs=10, validation_data=(X_test_pad, y_test), verbose=2)
history = model.fit(X_train_pad, y_train, batch_size=128, epochs=10, validation_data=(X_test_pad, y_test), verbose=2)

print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print('Testing...')
score, acc = model.evaluate(X_test_pad, y_test, batch_size=128)

print('Test score:', score)
print('Test accuracy:', acc)

print("Accuracy: {0:.2%}".format(acc))

filename = 'Model_save.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)

tao=0.5
b=1
predictions = []
while b!=0|(len(y_train))<=41996:
    model = pickle.load(open('Model_save.pkl', 'rb'))
    b = 0
    yy_predict = model.predict(X_unlabeled_pad)
    yy_predictt = yy_predict.tolist()
    predictions.append(yy_predictt)
    probability=model.predict_proba(X_unlabeled_pad)
#ll,aa = model.evaluate()
#    predictions_matrix = np.array(predictions)
#    daya = pd.DataFrame(predictions_matrix)
    for i in range(len(probability)):
        if probability[i][0]>tao:
            b=b+1
#            X_train_pad.append(X_train_pad,X_unlabeled_pad[i],axis=0)
            X_train_pad = np.concatenate((X_train_pad,X_unlabeled_pad[i].copy()[np.newaxis,:]),axis=0)
            if predictions[0][i][0]>0.5:
                y_train= np.append(y_train,1)
            else:
                y_train=np.append(y_train,0)

    history = model.fit(X_train_pad, y_train, batch_size=128, epochs=5, validation_data=(X_test_pad, y_test), verbose=2)
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    
score, acc = model.evaluate(X_test_pad, y_test, batch_size=128)

print('Test score:', score)
print('Test accuracy:', acc)

print("Accuracy of self training: {0:.2%}".format(acc))