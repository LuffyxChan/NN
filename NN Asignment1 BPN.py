# -*- coding: utf-8 -*-
"""Copy of artificial_neural_network.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Yi-i4aWEAwu3v9ND1puvKqwupH5vhJaH
"""

import numpy as np
import pandas as pd
import tensorflow as tf

dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0, shuffle = 1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=4, activation='relu'))

ann.add(tf.keras.layers.Dense(units=5, activation='relu'))

ann.add(tf.keras.layers.Dense(units=4, activation='relu'))

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

ann.fit(X_train, y_train, batch_size = 10, epochs = 100)

print(ann.predict([[0.5, 0.5]]))

y_pred = ann.predict(X_test)
print(y_pred)

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
accuracy_score(y_test, y_pred)
