import numpy as np
import matplotlib.pyplot as plt
import copy
import random
from imageio import imread


def IsScalar(x):
    if type(x) in (list, np.ndarray,):
        return False
    else:
        return True

def Thresh(x):
    if IsScalar(x):
        val = 1 if x>0 else -1
    else:
        val = np.ones_like(x)
        val[x<0] = -1.
    return val

def Hamming(x, y):
    d = []
    for xx, yy in zip(x,y):
        dd = 0.
        for xxx,yyy in zip(xx,yy):
            if xxx==1 and yyy!=1:
                dd += 1.
            elif yyy==1 and xxx!=1:
                dd += 1.
        d.append(dd)
    return d

def Perturb(x, p):
  y = copy.deepcopy(x)
  for yy in y:
    k = len(yy)
    deck = list(range(0, k))
    random.shuffle(deck)
    for h in range(p):
      i = deck[h]
      yy[i] *= -1
  return y

def Energy(W, b, X):
    E = []
    for xx in X:
        blah = - 0.5 * (xx @ W)@xx.T + b@xx.T
        E.append(blah)
    return E

def Update(W, x, b):
    xnew = x @ W - b
    return Thresh(xnew)

letters = []
letters.append(imread('0.png'))
letters.append(imread('1.png'))
letters.append(imread('2.png'))
letters.append(imread('3.png'))
letters.append(imread('4.png'))
letters.append(imread('6.png'))
letters.append(imread('#.png'))
letters.append(imread('9.png'))
n = len(letters)
N = len(letters[0].flatten())
X = np.zeros((n, N))
for idx,img in enumerate(letters):
    X[idx,:] = Thresh(np.array([img.flatten()-0.5]))

#plt.figure(figsize=(120,8))
for k in range(n):
    plt.subplot(1,n,k+1);
    plt.imshow(np.reshape(X[k], (12,10)), cmap='gist_gray'); plt.axis('off');

b = np.zeros((1,N))
b = np.sum(X, axis=0) / n
W = ( X.T @ X ) / n - np.eye(N)
W0 = copy.deepcopy(W)

k = np.random.randint(n)
Y = Perturb( X , p=30)
x = Y[k:k+1,]
#x[0,24:] = -1.
err = Hamming(x, X[k:k+1,:])
datalist = ['0','1','2','3','4','6','#','9']
print('Given Pattern to recognise: Pattern '+datalist[k]+' with '+str(err)+' errors')
x_orig = copy.deepcopy(x)

plt.imshow(np.reshape(x,[12,10]), cmap='gist_gray'); plt.axis('off');

print(W)
plt.subplot(121); plt.imshow(W, cmap='gist_gray'); plt.axis('off');

xs = copy.deepcopy(x_orig)
xa = copy.deepcopy(x)

print("Synchronous updating energey:")
n_iters = 5
for idx in range(n_iters):
    xs = Update(W, xs, b)
    e1 = Energy(W, b, xs)
    print(e1)

print("Asynchronous updating energey:")
n_iters = 5
for count in range(n_iters):
    e2 = Energy(W, b, xa)
    node_idx = list(range(N))
    np.random.shuffle(node_idx)
    for idx in node_idx:
        ic = xa@W[:,idx] - b[idx]
        xa[0,idx] = Thresh(ic)
    print(e2)

plt.subplot(1,2,2); plt.imshow(np.reshape(xa,[12,10]), cmap='gist_gray'); plt.title('Asynchronous'); plt.axis('off')
plt.subplot(1,2,1); plt.imshow(np.reshape(xs,[12,10]), cmap='gist_gray'); plt.title('Synchronous'); plt.axis('off')


print('Correct pattern is '+datalist[k])
print('Synchronous updating')
for idx,t in enumerate(X):
    ds = Hamming(xs, [t])[0]
    print('Memory pattern '+datalist[idx]+' has error '+str(ds))
print('Asynchronous updating')
for idx,t in enumerate(X):
    da = Hamming(xa, [t])[0]
    print('Memory pattern '+datalist[idx]+' has error '+str(da))

#To store image from array
#Source: Google
'''
import imageio
import numpy as np

array = np.array([-1, -1, -1, -1, -1, -1, -1, -1, 1, 1,
               -1, -1, -1, -1, -1, -1, -1, -1, 1, 1,
               1, 1, 1, 1, 1, 1, -1, -1, 1, 1,
               1, 1, 1, 1, 1, 1, -1, -1, 1, 1,
               1, 1, 1, 1, 1, 1, -1, -1, 1, 1,
               -1, -1, -1, -1, -1, -1, -1, -1, 1, 1,
               -1, -1, -1, -1, -1, -1, -1, -1, 1, 1,
               -1, -1, 1, 1, 1, 1, 1, 1, 1, 1,
               -1, -1, 1, 1, 1, 1, 1, 1, 1, 1,
               -1, -1, 1, 1, 1, 1, 1, 1, 1, 1,
               -1, -1, -1, -1, -1, -1, -1, -1, 1, 1,
               -1, -1, -1, -1, -1, -1, -1, -1, 1, 1])
array = np.reshape(array, (12, 10))
imageio.imwrite('2.png', array)'''
