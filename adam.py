import numpy as np
import math

X = [0.5,2.5]
Y = [0.2,0.9]
def f(w,x,b):
    return (1/(1+np.exp(-(w*x+b))))

def error(w,b):
    err=0.0
    for x,y in zip(X,Y):
        fx = f(w,x,b)
        err += 0.5*((fx-y)**2)
    return err

def grad_w(x,y,w,b):
    fx = f(w,x,b)
    return fx*(fx-y)*(1-fx)

def grad_b(x,y,w,b):
    fx = f(w, x, b)
    return fx * (fx - y) * (1 - fx)*x

def adam(x,y,max_epochs):
    lr =0.001
    w = 2
    b = 1
    v_w=0
    v_b=0
    eps = 0.9
    beta1 = 0.9
    beta2 = 0.999
    m_w=0
    m_b=0
    for i in range(max_epochs):
        dw=0
        db=0
        for xi,yi in zip(X,Y):
            dw += grad_w(xi,yi,w,b)
            db += grad_b(xi,yi,w,b)

        m_w = beta1*m_w + (1-beta1)*dw
        m_b = beta1 * m_b + (1 - beta1) * db

        v_w = beta2*v_w + (1-beta2)*dw**2
        v_b = beta2*v_b + (1-beta2)*db ** 2

        m_w = m_w / (1-beta1**i+1)
        m_b = m_b / (1-beta1**i+1)

        v_w = v_w / (1-beta2**i+1)
        v_b = v_b / (1-beta2**i+1)

        w = w - (lr/np.sqrt(v_w+eps))*m_w
        b = b - (lr / np.sqrt(v_b + eps)) * m_b

        print("Loss after {} epoch is {}".format(i, error( w, b)))

adam(X,Y,500)