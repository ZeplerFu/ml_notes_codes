import numpy
import numpy as np


def polyreg(x,y,lmd,xT=None,yT=None):
    #
    # Finds a D-1 order polynomial fit to the data
    #
    #    function [err,model,errT] = polyreg(x,y,D,xT,yT)
    #
    # x = vector of input scalars for training
    # y = vector of output scalars for training
    # D = the order plus one of the polynomial being fit
    # xT = vector of input scalars for testing
    # yT = vector of output scalars for testing
    # err = average squared loss on training
    # model = vector of polynomial parameter coefficients
    # errT = average squared loss on testing
    #
    # Example Usage:
    #
    # x = 3*(rand(50,1)-0.5);
    # y = x.*x.*x-x+rand(size(x));
    # [err,model] = polyreg(x,y,4);
    #
    x = np.matrix(x)
    y = np.matrix(y)
    xT = np.matrix(xT)
    yT = np.matrix(yT)
    model = (x.T*x + lmd*np.identity(x.shape[1])).I * x.T * y
    err = 1/(2*x.shape[0]) * np.linalg.norm(y-x*model)**2+lmd/(2*x.shape[0])*(np.linalg.norm(model)**2)
    if xT is not None:
        errT = 1/(2*xT.shape[0]) * np.linalg.norm(yT-xT*model)**2+lmd/(2*xT.shape[0])*(np.linalg.norm(model)**2)

    # Y = xT*model
    # Yy = [[y[i][0],numpy.array(Y[i][0])[0]] for i in range(len(Y))]
    # Yy = sorted(Yy)
    # y = [Yy[i][0] for i in range(len(Yy))]
    # Y = [Yy[i][1] for i in range(len(Yy))]
    # XX = range(0,len(Yy))
    # import matplotlib.pyplot as plt
    # if xT is None:
    #     plt.plot(XX,Y,'o',label='Y_Fitted')
    #     plt.plot(XX,y,'x',label='Y')
    #     plt.xlabel("Y-Index")
    #     plt.ylabel("Y-Value")
    #     plt.legend()
    #     plt.show()
    return err,errT
    # print(Y)
    # print(model)

def shuffle_split(x, y):
    # shuffle split into two equal size set
    xy = [[x[i],y[i]] for i in range(len(x))]
    import random
    random.shuffle(xy)
    # print(xy)
    xy_Train = xy[:int(len(x)//3)*2]
    xy_Test = xy[int(len(x)//3)*2:]
    X_Train = np.array([xy_Train[i][0] for i in range(len(xy_Train))])
    Y_Train = np.array([xy_Train[i][1] for i in range(len(xy_Train))])
    X_Test = np.array([xy_Test[i][0] for i in range(len(xy_Test))])
    Y_Test = np.array([xy_Test[i][1] for i in range(len(xy_Test))])
    return X_Train, Y_Train, X_Test, Y_Test

def shuffle_split_random():
    # shuffle split into two random size set
    pass

def load_data():
    import scipy.io
    Dataset_MATLAB = scipy.io.loadmat('problem2.mat')
    X_Data = Dataset_MATLAB['x']
    Y_Data = Dataset_MATLAB['y']
    return X_Data,Y_Data

if __name__ == "__main__":
    X_Data, Y_Data = load_data()
    # print(X_Data, Y_Data)
    X = np.array(X_Data)
    Y = np.array(Y_Data)
    # print(X.shape)
    # print(Y.shape)
    X_Train, Y_Train, X_Test, Y_Test = shuffle_split(X_Data,Y_Data)
    # X_Train.shape=(300,100)
    errors = []
    errorTs = []
    # # D = 4
    # polyreg(X_Data, Y_Data, 303)
    lmd = 1000
    for i in range(lmd):
        error, errorT = polyreg(X_Train,Y_Train,i,X_Test, Y_Test)
        errors.append(error)
        errorTs.append(errorT)
    index_min = numpy.argmin(errorTs)
    # print(errors)
    # print(errorTs)
    import matplotlib.pyplot as plt
    plt.plot(np.arange(0,lmd,1), errors, 'r', label="train")
    plt.plot(np.arange(0,lmd,1), errorTs, 'g', label="test")
    plt.legend()
    plt.title("CV Error")
    plt.show()
    print(index_min)
    #
    # print(index_min)
    print(errorTs)

