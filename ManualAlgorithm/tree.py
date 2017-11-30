import pandas as pd
import numpy as np

class Tree(object):
    def __init__(self):
        pass

    def __cal_entropy(self,x,y,solver='entropy'):
        #entropy
        #entropyrate
        #gini

        x = np.array(x)
        y = np.array(y)
        label_unique = np.unique(y)

        assert np.sum(np.unique(y)) == 1,"检查输入的标签列是否为0和1"
        m = len(x)
        m1 = np.sum(y)
        m0 = m - m1
        hd = (-m1/m) * np.log2(m1/m) + (-m0/m) * np.log2(m0/m)

        return hd

a dsf
class DecitionTree(Tree):

    pass



