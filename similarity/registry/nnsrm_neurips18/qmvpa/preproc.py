from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def scale_data(X_train, X_test):
    """
    X_train: (n_features, n_examples)
    scale axis 0 - feature normalization
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.T).T
    X_test = scaler.transform(X_test.T).T
    return X_train, X_test


def group_train_test_split(Xs, Y, test_prop, random_state):
    """train test split for a list of Xs
        Y is common to all Xs
    """
    Xs_train = []
    Xs_test = []
    for s in range(len(Xs)):
        X_train, X_test, y_train, y_test = train_test_split(
            Xs[s].T, Y, test_size=test_prop, random_state=0)
        X_train = X_train.T
        X_test = X_test.T
        # X_train, X_test = scale_data(X_train, X_test)
        Xs_train.append(X_train)
        Xs_test.append(X_test)
    return Xs_train, Xs_test
