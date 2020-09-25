from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split



iris = load_iris()

x = iris.data
y = iris.target

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=123)

clf = DecisionTreeClassifier(max_depth = 7, max_features = 2)

m = clf.fit(X_train, y_train)
predicted = clf.predict(X_test)

acc = accuracy_score(y_test, predicted)


print("Accuracy Score:  ", acc)
print("Predicted: ", predicted)






