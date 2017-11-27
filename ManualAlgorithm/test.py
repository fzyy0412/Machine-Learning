from ManualAlgorithm import LinearModel
import numpy as np
a = np.array([[1,2],[3,4],[5,6],[7,8],[9,10]])
b = a[:,0]*2 + a[:,1]*4
# a = np.random.rand(10000)
# b = a*2
clf = LinearModel.LinearRegression()
clf.fit(a,b)
print(clf.feature_importance)
print(clf.predict([[1,2],[3,4]]))
