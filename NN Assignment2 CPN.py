import numpy as np
import math
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

dataset = pd.read_csv('data1.csv')
X = dataset.iloc[:, 0:2].values
y = dataset.iloc[:, 2:4].values

n = 10
alpha = 0.1
beta = 0.1
gamma = 0.1
delta = 0.1

for loop in range(1):  
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0, shuffle = 1)

  from sklearn.preprocessing import StandardScaler
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  u = np.random.uniform(0, 0.5, (2, n))
  v = np.random.uniform(0, 0.5, (2, n))
  w = np.random.uniform(0, 0.5, (n, 2))
  s = np.random.uniform(0, 0.5, (n, 2))

  p = len(X_train)
  q = len(X_test)



  class CPN:
    def winner(self, l, u, v, w, s, X_train, y_train):
      d = []
      for  j in range(n):
        x1 = 0
        y1 = 0
        for i in range(2):
          x10 = (X_train[l][i] - u[i][j]) ** 2
          y10 = (y_train[l][i] - v[i][j]) ** 2
          x1 += x10
          y1 += y10
        d1 = math.sqrt(x1 + y1)
        d.append(d1)
      for J in range(n):
        if d[J] == min(d):
          return J
      
    def update(self, l, u, v, w, s, X_train, y_train, j, y_trainS):
      for i in range(2):
        u[i][j] += alpha * (X_train[l][i] - u[i][j])
        v[i][j] += beta * (y_train[l][i] - v[i][j])
        w[j][i] += gamma * (X_train[l][i] - w[j][i])
        s[j][i] += delta * (y_train[l][i] - s[j][i])
        if (i % 2 == 0):
          y_trainS.append(s[j][i])
      return u, v, w, s, y_trainS

    def test(self, s, j, y_testS):
      y_testS.append(s[j][0])
      return y_testS

    def CrossValtest(self, v, j, y_testCVS):
      y_testCVS.append(v[0][j])
      return y_testCVS



  ob = CPN()
  epochs = 100
  loss1 = []
  nepochs = []
  accuracy = []

  for epoch in range(epochs):
    '''if (epoch % 20) == 0:
      alpha -= 0.2
      beta -= 0.2
      gamma -= 0.2
      delta -= 0.2'''
    acc = 0
    y_trainS = []
    y_trainS1 = []
    y_trainS2 = []
    y_train1 = []
    lossTraining = []
    h = 0
    for l in range(p):
      j = ob.winner(l, u, v, w, s, X_train, y_train)
      u, v, w, s, y_trainS = ob.update(l, u, v, w, s, X_train, y_train, j, y_trainS)
    
    y_trainS1 = np.array(y_trainS).reshape(1,p)
    y_trainS2 = y_trainS
    for i in range(p):
      h = y_train[i][0]
      y_train1.append(h)
      if y_trainS2[i] > 0.5:
        y_trainS2[i] = 1
      else:
        y_trainS2[i] = 0
      if y_trainS2[i] == h:
        acc += 1
    accuracy.append(acc/p)


    y_train1 = np.array(y_train1).reshape(1,p)
    bce = tf.keras.losses.BinaryCrossentropy( from_logits=False , reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE )
    lossT = bce(y_train1, y_trainS1)
    loss1.append(lossT)
    #print(loss.numpy())
    nepochs.append(epoch)

  lossTraining.append(lossT)
  print(lossT)
  loss1 = np.array(loss1)

  plt.plot(nepochs, loss1, color = 'r')
  plt.plot(nepochs, accuracy, color = 'g')
  plt.xlabel("epochs")
  plt.ylabel("Loss (red) & Accuracy (green)")
  plt.title("Training Results with 10 neurons")
  plt.show()
  print(acc/p)





  y_testS = []
  y_testCVS = []
  y_testCVS1 = []
  y_testS1 = []
  y_test1 = []
  hp = 0
  ob = CPN()
  loss1 = []

  for l in range(q):
    j = ob.winner(l, u, v, w, s, X_test, y_test)
    y_testS = ob.test(s, j, y_testS)
    y_testCVS = ob.CrossValtest(v, j, y_testCVS)


  y_testS1 = np.array(y_testS).reshape(1,q)
  y_testCVS1 = np.array(y_testCVS).reshape(1,q)
  for i in range(q):
    hp = y_test[i][0]
    y_test1.append(hp)
  y_test1 = np.array(y_test1).reshape(1,q)

  bce = tf.keras.losses.BinaryCrossentropy( from_logits=False , reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE )
  lossG = bce(y_test1, y_testS1)
  lossCV = bce(y_test1, y_testCVS1)
  loss1.append(lossG)
  print(lossG)
  print('-----')
  '''print(loss.numpy())
  print(lossCV.numpy())'''

'''lossGen = np.array(loss1)
lossTraining = np.array(lossTraining)
np.savetxt('imdataGeneralization.csv', lossGen, delimiter=',')
print(loss1)
print(lossTraining)
np.savetxt('imdataTest.csv', lossTraining, delimiter=',')'''
