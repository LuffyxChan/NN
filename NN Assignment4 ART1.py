import numpy as np
import imageio
import copy
import random
import matplotlib.pyplot as plt

rho = 0.3
n = 64
m = 20
bi = 1/(1+n)

b = np.random.uniform(bi, bi, (m, n))
t = np.random.uniform(1, 1, (n, m))


S = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 0, 0, 0, 
                0, 0, 1, 1, 1, 1, 0, 0, 
                0, 0, 0, 0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 0, 0, 0, 
                0, 0, 1, 1, 1, 1, 0, 0,  
                0, 0, 1, 1, 1, 1, 0, 0,  
                0, 0, 1, 1, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 
                  0, 0, 0, 0, 0, 0, 0, 0,  
                  0, 0, 1, 1, 1, 1, 0, 0, 
                  0, 0, 0, 0, 0, 0, 0, 1, 
                  0, 0, 0, 0, 0, 0, 0, 1, 
                  0, 0, 1, 1, 1, 1, 0, 0,  
                  0, 0, 0, 0, 0, 0, 0, 0,  
                  0, 0, 0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 
                  0, 0, 0, 0, 0, 0, 0, 0,  
                  0, 0, 1, 1, 1, 1, 1, 1, 
                  0, 0, 1, 1, 1, 1, 1, 1, 
                  0, 0, 1, 1, 1, 1, 1, 1, 
                  0, 0, 1, 1, 1, 1, 1, 1, 
                  0, 0, 0, 0, 0, 0, 0, 0,  
                  0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 
                  0, 0, 0, 0, 0, 0, 0, 0,  
                  0, 0, 1, 1, 1, 1, 0, 0, 
                  0, 0, 1, 1, 1, 1, 0, 0, 
                  0, 0, 1, 1, 1, 1, 0, 0, 
                  0, 0, 1, 1, 1, 1, 0, 0, 
                  0, 0, 0, 0, 0, 0, 0, 0,  
                  0, 0, 0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 
                  0, 0, 0, 0, 0, 0, 0, 0, 
                  0, 0, 1, 1, 1, 1, 1, 1, 
                  0, 0, 0, 0, 0, 0, 0, 0, 
                  0, 0, 0, 0, 0, 0, 0, 0,  
                  0, 0, 1, 1, 1, 1, 1, 1,  
                  0, 0, 0, 0, 0, 0, 0, 0,  
                  0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 
                  0, 0, 0, 0, 0, 0, 0, 0, 
                  0, 0, 1, 1, 1, 1, 1, 1, 
                  0, 0, 0, 0, 0, 0, 0, 0, 
                  0, 0, 0, 0, 0, 0, 0, 0,  
                  0, 0, 1, 1, 1, 1, 1, 1,  
                  0, 0, 1, 1, 1, 1, 1, 1,  
                  0, 0, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 
                  0, 0, 0, 0, 0, 0, 0, 0, 
                  0, 0, 1, 1, 1, 1, 1, 1, 
                  0, 0, 1, 0, 0, 0, 0, 0, 
                  0, 0, 1, 0, 0, 0, 0, 0,  
                  0, 0, 1, 1, 1, 1, 0, 0,  
                  0, 0, 0, 0, 0, 0, 0, 0, 
                  0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 1, 1, 1, 0, 0,   
                  0, 0, 1, 1, 1, 1, 0, 0, 
                  0, 0, 1, 1, 1, 1, 0, 0, 
                  0, 0, 0, 0, 0, 0, 0, 0, 
                  0, 0, 0, 0, 0, 0, 0, 0,  
                  0, 0, 1, 1, 1, 1, 0, 0,  
                  0, 0, 1, 1, 1, 1, 0, 0,  
                  0, 0, 1, 1, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0,   
                  0, 0, 0, 0, 0, 0, 0, 0,  
                  1, 1, 1, 0, 0, 1, 1, 1, 
                  1, 1, 1, 0, 0, 1, 1, 1,  
                  1, 1, 1, 0, 0, 1, 1, 1,   
                  1, 1, 1, 0, 0, 1, 1, 1,  
                  0, 0, 0, 0, 0, 0, 0, 0,   
                  0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0,   
                  0, 0, 0, 0, 0, 0, 0, 0,  
                  1, 1, 1, 0, 0, 1, 1, 1, 
                  1, 1, 1, 0, 0, 1, 1, 1,  
                  1, 1, 1, 0, 0, 1, 1, 1,   
                  1, 1, 1, 0, 0, 1, 1, 1,  
                  0, 0, 0, 0, 0, 1, 1, 1,   
                  0, 0, 0, 0, 0, 1, 1, 1],
              [0, 0, 1, 1, 1, 1, 0, 0,   
                  0, 0, 1, 1, 1, 0, 0, 0,   
                  0, 0, 1, 1, 0, 0, 0, 1,  
                  0, 0, 0, 0, 0, 0, 1, 1,   
                  0, 0, 0, 0, 0, 0, 1, 1,    
                  0, 0, 1, 1, 0, 0, 0, 1,    
                  0, 0, 1, 1, 1, 0, 0, 0,   
                  0, 0, 1, 1, 1, 1, 0, 0],
              [0, 0, 1, 1, 1, 1, 1, 1,
                  0, 0, 1, 1, 1, 1, 1, 1, 
                  0, 0, 1, 1, 1, 1, 1, 1, 
                  0, 0, 1, 1, 1, 1, 1, 1, 
                  0, 0, 1, 1, 1, 1, 1, 1, 
                  0, 0, 1, 1, 1, 1, 1, 1, 
                  0, 0, 0, 0, 0, 0, 0, 0,  
                  0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 1, 1, 1, 0, 0,
                  0, 0, 0, 1, 1, 0, 0, 0,
                  0, 0, 0, 1, 1, 0, 0, 0,
                  0, 0, 1, 0, 0, 1, 0, 0,
                  0, 0, 1, 0, 0, 1, 0, 0,
                  0, 0, 1, 1, 1, 1, 0, 0,
                  0, 0, 1, 1, 1, 1, 0, 0,  
                  0, 0, 1, 1, 1, 1, 0, 0],
              [0, 0, 1, 1, 1, 1, 0, 0,
                  0, 0, 0, 1, 1, 1, 0, 0,
                  0, 0, 0, 0, 1, 1, 0, 0,
                  0, 0, 0, 0, 0, 1, 0, 0,
                  0, 0, 1, 0, 0, 0, 0, 0,
                  0, 0, 1, 1, 0, 0, 0, 0,
                  0, 0, 1, 1, 1, 0, 0, 0,  
                  0, 0, 1, 1, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 
                  0, 0, 1, 1, 1, 1, 0, 0, 
                  0, 0, 1, 1, 1, 1, 0, 0, 
                  0, 0, 1, 1, 1, 1, 0, 0, 
                  0, 0, 1, 1, 1, 1, 0, 0, 
                  0, 0, 0, 0, 0, 0, 0, 0,  
                  0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 
                  0, 0, 0, 0, 0, 0, 0, 0, 
                  0, 0, 1, 1, 1, 1, 0, 0, 
                  0, 0, 0, 0, 0, 0, 0, 0, 
                  0, 0, 0, 0, 0, 0, 0, 0,  
                  0, 0, 1, 1, 1, 1, 1, 1,  
                  0, 0, 1, 1, 1, 1, 1, 1,  
                  0, 0, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 
                  0, 0, 1, 1, 1, 1, 0, 0, 
                  0, 0, 1, 0, 0, 1, 0, 0, 
                  0, 0, 1, 0, 0, 0, 0, 0, 
                  0, 0, 1, 1, 0, 0, 0, 0, 
                  0, 0, 0, 0, 0, 0, 0, 0,  
                  0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 
                  0, 0, 0, 0, 0, 0, 0, 0, 
                  0, 0, 1, 1, 1, 1, 0, 0, 
                  0, 0, 0, 0, 0, 0, 0, 0, 
                  0, 0, 0, 0, 0, 0, 0, 0,  
                  0, 0, 1, 1, 1, 0, 0, 1,  
                  0, 0, 1, 1, 1, 0, 0, 0,  
                  0, 0, 1, 1, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 
                  0, 0, 0, 0, 0, 0, 0, 0, 
                  0, 0, 1, 1, 1, 1, 1, 1, 
                  0, 0, 0, 0, 0, 0, 0, 0, 
                  0, 0, 0, 0, 0, 0, 0, 0,  
                  1, 1, 1, 1, 1, 1, 0, 0,  
                  0, 0, 0, 0, 0, 0, 0, 0, 
                  0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0,   
                  0, 0, 0, 0, 0, 0, 0, 0,  
                  1, 1, 1, 0, 0, 1, 1, 1, 
                  1, 1, 1, 0, 0, 1, 1, 1,  
                  1, 1, 1, 0, 0, 1, 1, 1,   
                  1, 1, 1, 0, 0, 1, 1, 1,  
                  1, 1, 1, 0, 0, 1, 1, 1,   
                  1, 1, 1, 0, 0, 1, 1, 1]])

OP = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T']

plen = len(S)


def sumS(p):
  sumS1 = 0
  for i in range(n):
      sumS1 += S[p][i]
  return sumS1

def updateO(p,b,j):
  O = []
  for i in range(20):
    O.append(0)
  for p1 in range(plen):
    if p1 not in j:
      O1 = 0
      for i in range(n):
        O1 += S[p][i]*b[p1][i]
      O.pop(p1)
      O.insert(p1, O1)
    if p1 in j:
      O1 = -1
      O.pop(p1)
      O.insert(p1, O1)
  return O

def updateOT(test,b):
  O = []
  for i in range(20):
    O.append(0)
  for p in range(plen):
    O1 = 0
    for i in range(n):
      O1 += test[i]*b[p][i]
    O.pop(p)
    O.insert(p, O1)
  return O

def winner(O):
  for j in range(plen):
    if O[j] == max(O):
      J = j
      return J

def sumI(p,J):
  I = []
  I1 = 0
  sumI = 0
  for i in range(n):
    I1 = S[p][i]*t[i][J]
    sumI += I1
    I.append(I1)
  return I, sumI

def updateIT(test,J):
  I = []
  I1 = 0
  for i in range(n):
    I1 = test[i]*t[i][J]
    I.append(I1)
  return I

def checkV(sumS, sumI):
  if (sumI/sumS) > rho or (sumI/sumS) == rho:
    return True
  else:
    return False

def updateW(J,b,t,I,sumI2):
  for i in range(n):
    b[J][i] = (I[i])/(0.5 + sumI2)
    t[i][J] = I[i]
  return b,t

def train(p,b,t):
  j = []
  for k in range(plen):
    sumS2 = sumS(p)
    O = updateO(p,b,j)
    J = winner(O)
    I, sumI2 = sumI(p,J)
    Con = checkV(sumS2, sumI2)

    if Con == True:
      b,t = updateW(J,b,t,I,sumI2)
      break
    if Con == False:
      if len(j) != 20:
        j.append(J)
      else:
        print("pattern"+str(p)+"can not be clusttured.")
      
      continue
  return(b,t)

def addError(s, e):
  y = copy.deepcopy(s)
  k = len(y)
  deck = list(range(0, k))
  random.shuffle(deck)
  for h in range(e):
    i = deck[h]
    if y[i] == 1:       
      y[i] = 0
    else:
      y[i] = 1
  return y

def countError(s,ss):
  error = 0
  for i in range(n):
    if s[i] != ss[i]:
      error += 1
  return error


epoch = 1
for epochs in range(epoch):
  for p in range(plen):
    b,t = train(p,b,t)

E = []
test = S[h]

O = updateOT(test,b)
J = winner(O)
I = updateIT(test,J)


for i in range(plen):
  e = countError(S[i], I)
  E.append(e)
  #print('Error with letter '+str(OP[i])+' is '+str(e))

correct = E.index(min(E))
print('Correct image : '+str(OP[h]))
print('Recognised & Updated image has lowest error of '+str(min(E)) +' with letter '+str(OP[correct]))
plt.subplot(1,2,2); plt.imshow(np.reshape(I,[8,8]), cmap='gist_gray'); plt.title('Updated Image'); plt.axis('off')




plt.subplot(1,2,2); plt.imshow(np.reshape(S[19],[8,8]), cmap='gist_gray'); plt.title('Original Image'); plt.axis('off')

plt.subplot(1,2,2); plt.imshow(np.reshape(test,[8,8]), cmap='gist_gray'); plt.title('Noisy Image'); plt.axis('off')

plt.subplot(1,2,2); plt.imshow(np.reshape(I,[8,8]), cmap='gist_gray'); plt.title('Updated Image'); plt.axis('off')


#To store image from array
#Source: Google

import imageio
import numpy as np

array = np.array([0, 0, 0, 0, 1, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 1, 0, 0, 1, 0, 0,
                  0, 0, 1, 1, 1, 1, 0, 0,
                  0, 0, 1, 1, 1, 1, 0, 0,
                  0, 0, 1, 1, 1, 1, 0, 0,  
                  0, 0, 1, 1, 1, 1, 0, 0])
array = np.reshape(array, (8, 8))


plt.subplot(1,2,2); plt.imshow(np.reshape(array,[8,8]), cmap='gist_gray'); plt.title('Updated Image'); plt.axis('off')
