import pandas as pd
import numpy as np

class BaseTree(object):
    def __init__(self):
        pass
    def __cal_entropy(self,x,y):
        #entropy
        #entropyrate
        #gini

        x = np.array(x)
        y = np.array(y)
        label_unique = np.unique(y)

        assert np.sum(label_unique) == 1,"检查输入的标签列是否为0和1"
        m = len(x)
        m1 = np.sum(y)
        m0 = m - m1
        hd = (-m1/m) * np.log2(m1/m) + (-m0/m) * np.log2(m0/m)
        return hd






class DecisionTree(object):
    def __init__(self,max_depth = 3,solver='entropy'):

        self.max_depth = max_depth
        self.solver = 'entropy'

    def __cal_entropy(self,x,y):
        #entropy
        #entropyrate
        #gini

        x = np.array(x)
        y = np.array(y)
        label_unique = np.unique(y)

        assert np.sum(label_unique) == 1,"检查输入的标签列是否为0和1"
        m = len(x)
        m1 = np.sum(y)
        m0 = m - m1
        hd = (-m1/m) * np.log2(m1/m) + (-m0/m) * np.log2(m0/m)

        return hd

    def fit(self,x,y):
        __data_hd = self.__cal_entropy(x,y)



