from nltk.util import ngrams

class NGramFilter():
    
    def __init__(self):
        pass

    @staticmethod
    def ngrams(tokens, ngram) :
        return list(ngrams(tokens,ngram))