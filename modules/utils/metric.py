from sklearn.metrics import confusion_matrix

class Metric():
    
    def __init__(self):
        pass
    
    @staticmethod
    def confusion_matrix(y_pred, y_test):
        return confusion_matrix(y_test, y_pred)