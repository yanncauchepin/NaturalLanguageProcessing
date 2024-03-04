import numpy as np

class Utils():
    
    def __init__(self):
        pass
    
    @staticmethod 
    def softmax(vector):
        return np.exp(vector)/np.sum(np.exp(vector))
