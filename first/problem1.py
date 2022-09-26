import numpy as np


def polyreg(x,y,D,xT=None,yT=None):
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
    import numpy
    xx = np.ones((len(x),D),dtype=float)

    # for i in range(1,D):
    #     for j in range(len(x)):
    #         xx[j][i] = xx[j][i-1] * x[j]
    for i in range(0, D):
        xx[:,i] = x**(D-i-1)
    model = numpy.dot(numpy.linalg.pinv(xx),y)
    err = (1/(2*len(x)))*sum((y - numpy.dot(xx,model))**2)
    if xT is not None:
        xxT = np.ones((len(xT),D),dtype=float)
        for i in range(0,D):
            xxT[:,i]=xT**(D-i-1)
        errT = (1/(2*len(xT)))*sum((yT - numpy.dot(xxT,model))**2)
    q = numpy.arange(np.amin(x), np.amax(x), np.amax(x)/300.).T
    qq = np.zeros((q.shape[0],D), dtype=float)
    for i in range(0,D):
      qq[:,i] = q**(D-i - 1)
    # print(qq)
    # q  = (min(x):(max(x)/300):max(x))'
    Y = numpy.dot(qq,model)
    import matplotlib.pyplot as plt
    # if xT is not None:
    #     plt.plot(xT,yT,'o')
    # plt.plot(x,y,'x')
    # plt.plot(q,Y,'r')
    # plt.show()
    return err,errT
    # print(Y)
    # print(model)

def shuffle_split(x, y):
    # shuffle split into two equal size set
    xy = [[x[i][0],y[i][0]] for i in range(len(x))]
    import random
    random.shuffle(xy)
    print(xy)
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
    Dataset_MATLAB = scipy.io.loadmat('problem1.mat')
    X_Data = Dataset_MATLAB['x']
    Y_Data = Dataset_MATLAB['y']
    return X_Data,Y_Data

if __name__ == "__main__":
    X_Data, Y_Data = load_data()
    X = np.array(X_Data).ravel()
    Y = np.array(Y_Data).ravel()
    X_Train, Y_Train, X_Test, Y_Test = shuffle_split(X_Data,Y_Data)
    errors = []
    errorTs = []
    # D = 4
    for i in range(80):
        error, errorT = polyreg(X_Train,Y_Train,i,X_Test, Y_Test)
        errors.append(error)
        errorTs.append(errorT)
    import matplotlib.pyplot as plt
    plt.plot(np.arange(0,80,1), errors, 'r', label="train")
    # plt.title("CV Train Error")
    plt.plot(np.arange(0,80,1), errorTs, 'g', label="test")
    plt.title("CV Error")
    plt.legend()
    plt.show()

    print(errors)
    print(errorTs)
    # polyreg(X_Train,Y_Train,8,X_Test, Y_Test)


