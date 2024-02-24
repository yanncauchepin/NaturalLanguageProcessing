from sklearn.preprocessing import MinMaxScaler
"""min_max_standardization: between 0 and 1"""
def min_max_standardization():
    min_max_scaler = MinMaxScaler()
    '''df = min_max_scaler.fit_transform(df)'''
    

from sklean.preprocessing import StandardScaler
'''z_score_standardization: gauss distribution mostly between -3 and 3'''
def z_score_standardization():
    z_score_scaler = StandardScaler()
    '''df = z_score_scaler.fit_transform(df)'''
    
