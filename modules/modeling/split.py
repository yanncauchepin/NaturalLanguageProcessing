from sklearn.model_selection import train_test_split

class Split():
    
    def __init__(self):
        pass
    
    @staticmethod
    def standard(X, y, test_rate, stratify):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = test_rate, stratify = stratify)
        return {
            'X_train' : X_train,
            'X_test' : X_test,
            'y_train' : y_train,
            'y_test' : y_test
            }