from ManualAlgorithm import LinearModel
import numpy as np
a = np.array([[1,2],[3,4],[5,6],[7,8],[9,10],[11,12]])
b = [0,0,0,1,1,1]
# a = np.random.rand(10000)
# b = a*2
clf = LinearModel.LogisticRegression()
clf.fit(a,b)
#print(clf.feature_importance)
print(clf.predict([[5,11],[12,13]]))
