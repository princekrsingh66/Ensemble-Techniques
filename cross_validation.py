from sklearn import datasets, linear_model
from sklearn.model_selection import cross_validate
# loading diabetes data
diabetes = datasets.load_iris()
X = diabetes.data
y = diabetes.target
# calling lasso model
lasso = linear_model.Lasso()
# Cross validation on of model
results = cross_validate(lasso, X, y, cv=3)
print(results['test_score'])
