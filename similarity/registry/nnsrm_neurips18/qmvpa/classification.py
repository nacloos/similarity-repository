from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, LinearSVC
import numpy as np


def tune_lsvc(X_train, y_train, param_grid=None):
    if param_grid == None:
        param_grid = {
            'C': np.logspace(-5, 4, 10)
        }
    # tune SVM
    tuning_svm = LinearSVC(
        class_weight='balanced'
    )
    tuning_grid = GridSearchCV(
        estimator=tuning_svm, param_grid=param_grid, n_jobs=-1
    )
    tuning_grid.fit(X_train, y_train)
    return tuning_grid.best_estimator_, tuning_grid

# from sklearn.metrics import confusion_matrix
# cf_mat = confusion_matrix(Ys_test_srm_stkd,
#                           final_svm.predict(Xs_test_srm_stkd))
# plt.imshow(cf_mat, cmap='viridis')
