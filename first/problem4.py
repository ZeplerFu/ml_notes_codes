import math

import numpy as np

def cost_function(theta, y, fx):
    # y is true label
    # fx is predicted label
    # print(y)
    loss = 1/y.shape[0]*sum((y-1).T*np.log(1.-fx)-y.T*np.log(fx))
    # print(type(loss))
    return loss.item(0)

def update_model(model, lr, tol, x, fx, y):
    new_model = model - lr*(1/len(model)*sum((fx-y)*x))
    return new_model

def logi_reg(x,y,lr,tol,xT=None,yT=None):
    #
    # Finds a linear logstic fit to the data
    #
    #    function [err,model,errT] = polyreg(x,y,D,xT,yT)
    #
    # x = vector of input vectors for training
    # y = vector of output scalars for training
    # lr = learning rate(eta)
    # tol = tolerance of theta(epsilon)
    # xT = vector of input vectors for testing
    # yT = vector of output scalars for testing
    # err = loss on training
    # model = vector of polynomial parameter coefficients(theta vector)
    # errT = loss on testing
    #
    # Example Usage:
    #
    # x = 3*(rand(50,1)-0.5);
    # y = x.*x.*x-x+rand(size(x));
    # [err,model] = polyreg(x,y,4);
    #
    # x.shape=(200,3)
    orx = x
    ory = y
    x = np.matrix(x, dtype=float)
    y = np.matrix(y,dtype=float)
    xT = np.matrix(xT,dtype=float)
    yT = np.matrix(yT,dtype=float)
    # print("xshape:",x.shape)
    model = np.matrix(np.random.random(x.shape[1]),dtype=float).T
    # model.shape=(3,1)
    fx = 1/(1+np.exp(-1*x*model))
    # fx.shape=(200,1)
    err = cost_function(model, y, fx)
    new_model = model - lr*(1/len(model)*(x.T*(fx-y)))
    # print("!!!!")
    indexList = []
    errList = []
    t = 1
    while np.linalg.norm(new_model-model) >= tol:
        # print(np.linalg.norm(new_model-model))
        model = new_model
        fx = 1/(1+np.exp(-1*x*model))
        new_model = model - lr*(1/len(model)*(x.T*(fx-y)))
        err = cost_function(model, y, fx)
        # print("err:{} norm:{}".format(err, np.linalg.norm(new_model-model)))
        errList.append(err)
        indexList.append(t)
        t += 1
    x10 = []
    x11 = []
    x00 = []
    x01 = []
    for i in range(len(y)):
        if y[i] == 1:
            x10.append(orx[i][0])
            x11.append(orx[i][1])
        else:
            x00.append(orx[i][0])
            x01.append(orx[i][1])
    xx = np.arange(0,1,0.1)
    yy = (-1*(model.item(0)/model.item(1))*xx-(math.log(1)+model.item(2))/model.item(1))
    import matplotlib.pyplot as plt
    plt.plot(indexList, errList, 'r')
    plt.show()
    plt.plot(x00, x01, 'b.', label="y=0")
    plt.plot(x10, x11, 'g.', label="y=1")
    plt.plot(xx,yy,'r',label="decision")
    plt.legend()
    plt.show()
    cnt = 0
    for i in range(len(y)):
        if fx[i]>0.5 and 1 != y[i]:
            # print(fx[i], y[i])
            cnt+=1
    print(model)
    print(cnt)
    return err


def load_data():
    import scipy.io
    Dataset_MATLAB = scipy.io.loadmat('dataset4.mat')
    # print(Dataset_MATLAB)
    X_Data = Dataset_MATLAB['X']
    Y_Data = Dataset_MATLAB['Y']
    return X_Data,Y_Data

if __name__ == "__main__":
    X_Data, Y_Data = load_data()
    # print(X_Data, Y_Data)
    X = np.array(X_Data)
    Y = np.array(Y_Data)
    # print(X.shape, Y.shape)
    logi_reg(X,Y,0.0001, 0.00001)

