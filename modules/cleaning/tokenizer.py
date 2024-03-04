import nltk
# nltk.download('punkt') ISSUE
from nltk.tokenize import (
    word_tokenize,
    RegexpTokenizer,
    TreebankWordTokenizer,
    TweetTokenizer
    )

class Tokenizer():

    def __init__(self):
        pass

    @staticmethod
    def split(text) :
        return text.split()
    
    @staticmethod
    def standard(text) :
        """Not work"""
        return word_tokenize(text)
    
    @staticmethod    
    def customized_regex(regex, text) :
        regexp = RegexpTokenizer(regex)
        return regexp.tokenize(text)
    
    @staticmethod
    def blank_line(text) :
        """nltk BlankLine"""
        pass
    
    @staticmethod
    def word_punct(text) :
        """nltk WordPunct"""
        pass
    
    @staticmethod
    def treebank_word(text) :
        treebankword = TreebankWordTokenizer()
        return treebankword.tokenize(text)
    
    @staticmethod
    def tweet(text) :
        # can add parameters strip_handles=True, reduce_len=True, preserve_case=False
        tweet = TweetTokenizer()
        return tweet.tokenize(text)


if __name__ == '__main__' :

    """EXAMPLE"""

    text = "This is a test sentence for tokenization."

    tokens = Tokenizer.tokenizer(text)
    print("Tokenization output :", tokens)

    takens = Tokenizer.treebank_word_tokenizer(text)
    print("Tokenization output :", tokens)
