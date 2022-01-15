from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

def train_model(Model,x,y, N_folds = 5):
    """
    This function will help you to train your N - fold model. 
    Arguments
    --------------
    Model : scikit-learn model. E.g. DecisionTree()
    x : X_train dataframe(independent variables).
    y : y_train dataframe(target variable).
    N_folds : Number of folds for training.

    Return
    ---------
    Model : trained model on passed dataset.
    X_test : X_test dataframe(independent variables.
    y_test : y_test dataframe(target variable).  
    """
    x = x.values
    y = y.values.ravel()
    # preserving the percentage of samples for each class.
    skf = StratifiedKFold(n_splits= N_folds)

    for train_index, test_index in skf.split(x, y):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # fitting on model
        Model = Model.fit(X_train, y_train)
        y_hat = Model.predict(X_test)
        accuracy = accuracy_score(y_test, y_hat)
        precision = precision_score(y_test, y_hat,average='macro')
        recall = recall_score(y_test, y_hat,average='macro')
        f1 = f1_score(y_test, y_hat,average='macro')
        print("accuracy score",accuracy)
        print f"precision is {precision}| recall is {recall}| f1 score is {f1}"
    return Model, X_test, y_test
