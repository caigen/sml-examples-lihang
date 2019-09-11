import matplotlib.pyplot as plt

# =================================
# Data
from sklearn import datasets

iris = datasets.load_iris()
print("iris:")
print(iris.data)
print(iris.target)

digits = datasets.load_digits()
print("digits:")
print(digits.data)
print(digits.target)
print(digits.images[0])

plt.figure(1, figsize=(5, 5))
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()

# =================================
# Model
from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.0)
# Use all but the last item
clf.fit(digits.data[:-1], digits.target[:-1])
# Predict the last item
result = clf.predict(digits.data[-1:])

print("digits prediction:")
print(result[0])

# =================================
# Model persistence
clf = svm.SVC(gamma="scale")
X, y = iris.data, iris.target
clf.fit(X, y)

# [0] pickle
import pickle

s = pickle.dumps(clf)
clf2 = pickle.loads(s)
result = clf2.predict(X[0:1])

print("iris prediction:")
print("real: ", y[0])
print(result[0])

# [1] joblib
from joblib import dump, load
dump(clf, "tutorial.joblib")
clf3 = load("tutorial.joblib")
result = clf3.predict(X[0:1])
print(result[0])
