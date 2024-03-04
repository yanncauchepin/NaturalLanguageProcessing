import samples
from samples.classifier_imdb.preprocessing import ImdbPreprocessing
from samples.classifier_opinions.preprocessing import OpinionsPreprocessing

import modules
from modules.cleaning.tokenizer import Tokenizer
from modules.cleaning.stemmer import Stemmer
from modules.cleaning.stop_words_filter import StopWordsFilter
from modules.modeling.data_structure import DataStructure
from modules.modeling.split import Split

class Classifier():
    
    def __init__(self):
        pass
    
    
if __name__ == '__main__':
    
    """EXAMPLE"""
    
    """IMDB"""
    '''
    df_imdb = ImdbPreprocessing.load_dataframe()
    df_imdb= ImdbPreprocessing.df_parser(df_imdb, 'review')

    imdb_tokens = Tokenizer.split(df_imdb.loc[0, 'review'])
    imdb_tokens = Stemmer.porter(imdb_tokens)
    imdb_tokens = StopWordsFilter.english(imdb_tokens)
    print(imdb_tokens)
    '''
    
    """Opinions Amazon"""
    df_amazon = OpinionsPreprocessing.load_dataframe("amazon")
    bag_of_words = DataStructure.bag_of_words(df_amazon['texts'])
    feature_vectors = bag_of_words['feature_vectors']
    tfidf_feature_vectors = DataStructure.tfidf_feature_vectors(feature_vectors)
    dataset = Split.standard(tfidf_feature_vectors, df_amazon['opinions'], 0.2, df_amazon['opinions'])
    print(dataset)