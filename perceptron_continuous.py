import numpy as np
import itertools

def unipolar_activation(net,lambdaa):
    return (1/(1+np.exp(-lambdaa*net)))

def bipolar_activation(net,lambdaa):
    return (2/(1+np.exp(-lambdaa*net)))-1

def unipolar_fdash(y_pred):
    return (y_pred*(1-y_pred))

def bipolar_fdash(y_pred):
    return (1/2)*(1-(y_pred**2))

def calculate_r(yi,y_pred,type):
    r = yi - y_pred
    if type=='unipolar':
        return r*unipolar_fdash(y_pred)
    else:
        return r*bipolar_fdash(y_pred)

def predict(xi,yi,w,type):
    net = np.dot(xi,w)
    if type =='unipolar':
        return unipolar_activation(net,0.3)
    else:
        return bipolar_activation(net,1)


def train(X,Y,w,n,max_epochs,type):
    for i in range(max_epochs):
        loss=0
        for xi,yi in zip(X,Y):
            y_pred = predict(xi,yi,w,type)
            r = calculate_r(yi,y_pred,type)
            loss+=abs(r)
            delta_w = 0.5*r*xi
            w +=delta_w

        print("Loss after {}th epoch is {}".format(i,loss))

    w = w.reshape((n+1,1))
    test(X,w,type)

def test(X,lw,type):
    nets = np.dot(X,lw).flatten()
    # print('Actual Values : ',y)
    if type == 'unipolar':
        y_pred = np.array([unipolar_activation(net,0.3) for net in nets])
        print('Predicted Values : ', y_pred)
    else:
        y_pred = np.array([bipolar_activation(net,1) for net in nets])
        print('Predicted Values : ', y_pred)


n= int(input("Enter number of bits"))
X = np.array([list(i)+[1] for i in itertools.product([0,1],repeat=n)])

w = input(f'Enter initial {n} weights and 1 bias : ')
w = np.array([float(weight) for weight in w.split()], dtype='longdouble')
print()

#AND gate unipolar
Y= np.array([0]*(2**n))
Y[-1] = 1
print(Y)
print(Y.shape)
train(X,Y,w.copy(),n,500,'unipolar')


