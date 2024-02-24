from sklearn.preprocessing import LabelEncoder
"""Label encoder:
By using label encoding, we have addressed the issue of non-numeric data values 
in the dataset. However, encoding categorical variables that are nominal (where 
the values of the variable can't be ordered; for example, gender, days in a week, 
color, and so on) and not ordinal (the values of the variable can be ordered; for 
example, rank, size, and so on) creates another complication. For example, in the 
preceding case, we encoded Friday as 0 and Saturday as 1. When we feed these values 
to a mathematical model, it will consider these values as numbers and therefore will 
consider 1 to be greater than 0, which is not a correct treatment of these values."""
def label_encoder():
    label_encoding = LabelEncoder()
    '''column_fit = label_encoding.fit(df.[column])
    df.apply(label_encoding.fit_transform)'''
    
    
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
"""one-hot encoding:
There is still an outstanding issue that we need to resolve and that is the issue 
of the dummy variable trap. Say we use one-hot encoding on the day variable, which 
has four unique values. Splitting this variable into four columns will cause collinearity 
in our data (high correlation between variables) because we can always predict the 
outcome of the fourth column with the three other columns (if the day is not Friday, 
Saturday, or Sunday, then it will have to be Thursday). To address this issue, we 
will need to drop one dummy variable from the split columns of each categorical variable. 
This could be done by simply passing the argument drop='first' when defining the 
OneHotEncoder class."""
def one_hot_encoder():
    one_hot_encoding = ColumnTransformer([('OneHotEncoding', OneHotEncoder(),\
                                           [columns])], remainder='passthrough')
    '''df = one_hot_encoding.fit_transform(df)'''
    
