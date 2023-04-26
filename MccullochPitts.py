import numpy as np
import itertools

def predict(X,w,n,T):
    net = np.dot(X,w)
    print(net)
    y_pred = []
    for i in range(2**n):
        if net[i][0]>=T:
            y_pred.append(1)
        else:
            y_pred.append(0)
    print(y_pred)

#NOT
X = np.array([[0],[1]])
w = np.array([-1]).reshape((1,1))
T=0
predict(X,w,1,T)


n = int(input("Enter no. of bits"))
X = np.array([list(i) for i in itertools.product([0,1],repeat=n)])
# print(X.shape)

# AND gate
T = n
w = np.array([1]*n).reshape((n,1))
# print(w)
# print(w.shape)
predict(X,w,n,T)

# NAND gate
T = -n+1
w = np.array([-1]*n).reshape((n,1))
# print(w)
# print(w.shape)
predict(X,w,n,T)

# OR gate
T = 1
w = np.array([1]*n).reshape((n,1))
# print(w)
# print(w.shape)
predict(X,w,n,T)

# NOR gate
T = 0
w = np.array([-1]*n).reshape((n,1))
# print(w)
# print(w.shape)
predict(X,w,n,T)