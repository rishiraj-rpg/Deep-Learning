import numpy as np
import itertools

def unipolar_activation(net):
    if net>=0: return 1
    else: return 0

def bipolar_activation(net):
    if net>=0: return 1
    else: return -1

def predict(x,w,type):
    net= np.dot(x,w)
    if type=='unipolar':
        return unipolar_activation(net)
    else:
        return bipolar_activation(net)

def train(X,Y,n,w,type):
    for i in range(500):
        loss = 0
        for xi,yi in zip(X,Y):
            y_pred = predict(xi,w,type)
            r = yi-y_pred
            loss+=abs(r)
            delta_w = 0.5*r*xi
            w+=delta_w

        print("loss after {} epoch is {}".format(i,loss))
    w = w.reshape((n+1,1))
    test(X,w)

def test(X,lw):
    nets = np.dot(X,lw).flatten()
    print(nets)

n = int(input("Enter no. of bits"))
X = np.array([list(i) +[1] for i in itertools.product([0,1],repeat=n)])

w = input("Enter 2 weights and a bias")
w = np.array([float(weight) for weight in w.split()],dtype='longdouble')
Y = np.array([0]*(2**n))
Y[-1] = 1
train(X,Y,n,w.copy(),'unipolar')

