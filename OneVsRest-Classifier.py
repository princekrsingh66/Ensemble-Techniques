from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split

# loading iris and splitting data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
clf = RandomForestClassifier(n_estimators=10, random_state=42)

# fitting with RF model
clf.fit(X_train, y_train).score(X_test, y_test)
>>> .8947

# fitting with OneVsRestClassifier
clf2 = RandomForestClassifier(n_estimators=10, random_state=42)
OneVsRestClassifier(clf2).fit(X, y).score(X_test, y_test)
>>> 1.0
