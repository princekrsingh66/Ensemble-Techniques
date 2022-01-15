from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

# Select Ensemble models which would be best fit for stacking model
estimators = [('rf', RandomForestClassifier(n_estimators=10, random_state=42)),('adb', AdaBoostClassifier(n_estimators=100, random_state=0))]

# Stacking classifier
clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

# iris data
X, y = load_iris(return_X_y=True)

# splitting dataset and fitting on models
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

clf.fit(X_train, y_train).score(X_test, y_test)
