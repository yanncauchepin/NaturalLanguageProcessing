from sklearn.preprocessing import MinMaxScaler
from sklean.preprocessing import StandardScaler

class Scaler():
    
    def __init__(self):
        pass

    @staticmethod
    def min_max():
        """min_max_standardization: between 0 and 1"""
        min_max_scaler = MinMaxScaler()
        '''df = min_max_scaler.fit_transform(df)'''
        
    @staticmethod
    def z_score():
        '''z_score_standardization: gauss distribution mostly between -3 and 3'''
        z_score_scaler = StandardScaler()
        '''df = z_score_scaler.fit_transform(df)'''
        
