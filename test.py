from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris


iris = load_iris()

x = iris.data
y = iris.target


clf = DecisionTreeClassifier(max_depth = 7, max_features = 2)

m = clf.fit(x, y)
labeL_test = clf.predict(x)

acc = accuracy_score(labeL_test, y)
print("Accuracy Score:  ", acc)

print("Predicted: ", labeL_test)

