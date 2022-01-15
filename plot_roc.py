from sklearn import SVM
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle

model = SVM.SVC()
model.fit(X_test, y_test)

def ROC(y_test, x_test, model):
    """
    This function will help you to plot multiclass ROC curve provided that model supports decision_function.
    
    Arguments:
    ----------
    y_test, x_test in data array format.
    
    Returns:
    --------
    Graph of TPR vs FPR
    
    """
    y_score = model.decision_function(x_test)
    n_classes = np.unique(y_test)
    y_test = label_binarize(y_test, classes = np.unique(y_test))
    lw =2
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(n_classes)):
        fpr[n_classes[i]], tpr[n_classes[i]], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[n_classes[i]] = auc(fpr[n_classes[i]], tpr[n_classes[i]])

    plt.figure()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(n_classes, colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

ROC(y_test,X_test,model)
