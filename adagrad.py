import numpy as np
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

def adagrad(x,y,max_epochs):
    lr =0.001
    w = 2
    b = 1
    v_w=0
    v_b=0
    eps = 0.9
    for i in range(max_epochs):
        dw=0
        db=0
        for xi,yi in zip(X,Y):
            dw += grad_w(xi,yi,w,b)
            db += grad_b(xi,yi,w,b)

        v_w = v_w + dw**2
        v_b = v_b + db ** 2

        w = w - (lr/np.sqrt(v_w+eps))*dw
        b = b - (lr / np.sqrt(v_b + eps)) * db

        print("Loss after {} epoch is {}".format(i, error( w, b)))

adagrad(X,Y,500)