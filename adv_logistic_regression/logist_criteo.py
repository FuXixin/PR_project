from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import data_processing as dp
from sklearn.preprocessing import scale


def sigmoid(Z):
    return 1.0 / (1 + np.exp(-Z))

# 使用梯度下降法更新参数
def gradA(X, Y, k):
    m, n = X.shape
    # 学习率
    alpha = - 0.5
    # 权重
    b = 0
    W = np.ones((n, 1))  
    for i in range(k):
        Z = np.dot(W.T, X.T) + b
        A = sigmoid(Z)
        dZ = A - Y.T  
        # 每次迭代的更新值 
        dW = np.dot(X.T, dZ.T) / m  
        db = np.sum(dZ) / m
        L = -(np.dot(Y, np.log(A)) + np.dot(1 - Y, np.log(1 - A)))
        J = np.sum(L) / m   # logistic 的损失函数
        print(i, J)
        W = W + alpha * dW
        b = b + alpha * db
    return W, b

if __name__ == '__main__':

    d = pd.read_csv(r".\data\train.csv",usecols=range(1, 15))
    #print(d)
    d = np.array(d)
    # print(d.shape)
    x = d[:, 1:]
    y_train = d[:, 0].reshape(-1, 1)
    # print(x)
    # print(y_train)
    x = dp.insert_nan(x)
    
    x_train = dp.NL(x)
    # print(x_train)
    n = int(input("迭代次数："))
    W1, b1 = gradA(x_train, y_train, n)
    print(W1, b1, '\n')

    d = pd.read_csv(r".\data\test.csv", usecols=range(0, 14))
    d = np.array(d)
    x = d[:, 1:]
    Id = d[:, 0]
    x = dp.insert_nan(x)
    x_test = scale(x)
    pre_Y = sigmoid(W1.T.dot(x_test.T)+b1)

    pre_Y = pre_Y.tolist()[0]
    Id = Id.tolist()
    # 输出预测文件
    dataframe = pd.DataFrame({'Id': Id, 'Label': pre_Y})
    dataframe.to_csv("./data/result.csv", index=False, sep=',')
    # print(Y)
