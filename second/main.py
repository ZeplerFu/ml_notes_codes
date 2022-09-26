import numpy as np
import matplotlib.pyplot as plt
def load_data():
    import scipy.io
    Dataset_MATLAB = scipy.io.loadmat('data3.mat')
    Data = Dataset_MATLAB['data']
    X_Data = [Data[i][:-1] for i in range(len(Data))]
    Y_Data = [Data[i][-1] for i in range(len(Data))]
    return X_Data,Y_Data

def perceptron(X, Y, eta):
    w=np.ones((X.shape[1],1))
    X=np.matrix(X)
    Y=np.matrix(Y).T
    b=1
    round = 0
    all_correct=False
    while True:
        Errors = 0
        for i in range(X.shape[0]):
            XX = X[i]
            YY = Y[i]
            if ((w.T*XX.T+b).getA()[0][0] <0 and YY.getA()[0][0] >0) or ((w.T*XX.T+b).getA()[0][0] >0 and YY.getA()[0][0] <0):
                w+=eta*(XX.T*YY)
                b+=eta*YY
                Errors += 1
        if Errors == 0:
            break
        round += 1
        print(round, Errors)
    return w, b

def draw(w, b, X, Y):
    x_points = np.arange(0, 1, 0.1)
    y_points = np.array([-(w[0]*i+b)/w[1] for i in x_points]).flatten()
    x_0 = []
    x_1 = []
    for i in range(X.shape[0]):
        if Y[i] == -1:
            x_0.append(X[i])
        else:
            x_1.append(X[i])
    x_0 = np.array(x_0)
    x_1 = np.array(x_1)
    plt.plot(x_0[:, 0], x_0[:, 1], 'o', color='green', label='-1')
    plt.plot(x_1[:, 0], x_1[:, 1], 'o', color='yellow', label='1')
    plt.plot(x_points, y_points, '-', color='red')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    X_Data, Y_Data = load_data()
    X = np.array(X_Data)
    Y = np.array(Y_Data)
    w,b=perceptron(X, Y, 0.01)
    draw(w, b, X, Y)
