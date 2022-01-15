from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# setting all combination of params
parameters = {'n_estimators': range(10,400,50),'criterion':['gini', 'entropy'],'oob_score':[True, False],'bootstrap' :[True, False],'max_features':['auto', 'sqrt', 'log2'],'class_weight':['balanced', 'balanced_subsample']}
# grid code
grid = GridSearchCV(estimator = RandomForestClassifier(), param_grid=parameters,verbose=2, cv=5,n_jobs=-1)
# Fitting on grid
grid.fit(X_test, y_test)
# calling best fitted params
grid.best_estimator_
