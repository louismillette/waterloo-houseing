import numpy as np
import random
import statsmodels.api as sm
from statsmodels.formula.api import ols
from pandas import DataFrame

class lm():
    def __init__(self):
        pass
    def split(self, X, Y, split):
        data = [list(ele) for ele in zip(X,Y)]
        random.shuffle(data)
        train = data[:int(len(data) * split)]
        test = data[int(len(data) * split):]
        X_train = [ele[0] for ele in train]
        Y_train = [ele[1] for ele in train]
        X_test = [ele[0] for ele in test]
        Y_test = [ele[1] for ele in test]
        return X_train, Y_train, X_test, y_test
    # X data is an array of arrays
    def fit(self,X,Y):
        factors = len(X[0])
        data = [tuple([ele[0]]+ele[1]) for ele in zip(Y,X)]
        labels = [('y','int16')]+[('e{}'.format(ele),'int16') for ele in range(factors)]
        np_array = np.array(data, dtype=labels)
        expr = "y ~ " + '*'.join([ele[0] for ele in labels[1:]]) + ''
        print(expr)
        mod = ols(expr,data=np_array).fit()
        #aov_table = sm.stats.anova_lm(mod, typ=2)
        #print(aov_table)

l = lm()
l.fit(X=[[3,9,5],[6,7,2],[5,6,7]], Y=[1,2,3])

        #
        # mod = ols('weight ~ group',
        #           data=data).fit()
        #
        # aov_table = sm.stats.anova_lm(mod, typ=2)
        # print
        # aov_table

