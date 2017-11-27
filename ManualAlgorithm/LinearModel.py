import pandas as pd
import numpy as np
class LinearRegression(object):

    '''
    ----------线性回归算法----------
    1.lr：学习率
    默认0.001

    2.solver：求解方法
    normal正规方程，batchgrad批梯度下降，stochastic随机梯度下降

    3.eps：loop停止的误差阈值
    默认0.001
    '''

    def __init__(self,lr = 0.001,solver = 'batchgrad',eps = 0.001):

        self.lr = lr
        self.solver = solver
        self.eps = eps

    def fit(self,x,y):

        #处理单变量
        if len(x.shape)==1:
            x = x.reshape(-1, 1)

        #处理多变量
        self.M = x.shape[0]
        self.N = x.shape[1]
        y = y.reshape(-1, 1)

        #正规方程法求解
        if self.solver == 'normal':
            xt = np.transpose(x)
            self.coef = np.dot(np.dot(np.linalg.inv(np.dot(xt,x)),xt),y)

        #批梯度下降法求解
        if self.solver == 'batchgrad':
            w = np.array([np.random.randn() for i in range(self.N)])
            w = w.reshape(-1,1)

            loss = 10000
            while abs(np.average(loss)) > self.eps:
                y_hat = np.dot(x,w)
                loss = y_hat - y
                grad = np.sum(loss*x,0).reshape(-1,1)
                w = w - self.lr * grad
            self.coef = w

        #随机梯度下降
        if self.solver == 'stochastic':
            w = np.array([np.random.randn() for i in range(self.N)])
            w = w.reshape(-1,1)

            for m in range(self.M):
                y_hat = np.dot(x[m],w)
                loss = y_hat - y[m]
                grad = loss*m
                w = w - self.lr * grad
            self.coef = w

    def predict(self,x):
        x = np.array(x)

        if len(x.shape) == 1:
            x = x.reshape(-1,1)
            self.y_hat = self.coef * x
            return self.y_hat

        assert x.shape[1] == self.N,"输入维度错误"
        self.y_hat = np.dot(np.transpose(self.coef),x).ravel()
        return self.y_hat

    @property
    def feature_importance(self):
        return self.coef.ravel()

