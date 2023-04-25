import numpy as np

X = [0.5,2.5]
Y = [0.2,0.9]

def f(w,x,b):
    return 1/(1+np.exp(-(w*x+b)))

def error(x,y,w,b):
    err=0.0
    for x,y in zip(X,Y):
       fx = f(w,x,b)
       err+=0.5*((fx-y)**2)
    return err

def grad_w(x,y,w,b):
    fx = f(w,x,b)
    return fx*(fx-y)*(1-fx)

def grad_b(x,y,w,b):
    fx = f(w, x, b)
    return fx*(fx-y)*(1-fx)*x

def momentum(x,y,max_epoch):
    w = 2
    b = 1
    lr = 0.001
    gamma = 0.9
    v_w=0
    v_b=0
    prev_v_w=0
    prev_v_b=0
    for i in range(max_epoch):
        dw=0
        db=0
        v_w = gamma*prev_v_w
        v_b = gamma*prev_v_b
        for xi,yi in zip(X,Y):
            dw += grad_w(xi, yi, w-v_w, b-v_b)
            db += grad_b(xi, yi, w-v_w, b-v_b)
        v_w = gamma*prev_v_w + lr*dw
        v_b = gamma*prev_v_b + lr*db
        w = w - v_w
        b = b- v_b
        prev_v_w = v_w
        prev_v_b =v_b

        print("Loss after {} epoch is {}".format(i,error(x,y,w,b)))



momentum(X,Y,20)