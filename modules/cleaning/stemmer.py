import os
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

class Stemmer():    
    """Word stemming is the process of transforming a word into its root form.
    It allows us to map related words to the same stem.
    Potential problems arise in the form of over-stemming and under-stemming :
    - over-stemming : same root should have been stemmed to different roots
    - under-stemming : different roots should have been stemmed to same root
    Word stemming can create non-real words."""
    
    def __init__(self):
        pass
    
    @staticmethod
    def porter(tokens) :
        """The Porter stemming algorithm works only with string."""
        porter = PorterStemmer()
        return [porter.stem(token) for token in tokens]
    
    @staticmethod
    def snowball(tokens) :
        """The Snowball stemming algorithm (Porter2 or English stemmer) is faster 
        than the original Porter stemmer. It can work with both string and unicode 
        data."""
        # snowball stemmer requires the specification of a language parameter
        snowball = SnowballStemmer(language='english')
        return [snwoball(token) for token in tokens]
    
    @staticmethod
    def lancaster(tokens) :
        """The Lancaster stemming (Paice/Husk stemmer) is faster and more aggressive 
        than the original Porter stemmer : it produces shorter and more obscur words."""
        pass
    
    @staticmethod
    def dawson(tokens) :
        pass
    
    @staticmethod
    def krovetz(tokens) :
        pass
    
    @staticmethod
    def lovins(tokens) :
        pass


if __name__ == '__main__' :

    """EXAMPLE"""

    tokens = ["This", "is", "a", "test", "sentence", "with", "some", "stop", "words"]

    stemmed_tokens = Stemming.porter(tokens)
    print("Porter stemming output :", stemmed_tokens)
