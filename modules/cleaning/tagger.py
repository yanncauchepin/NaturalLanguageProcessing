import nltk
nltk.download('tagsets')
# nltk.help.upenn_tagset()
#nltk.download('averaged_perceptron_tagger')
from sutime import SUTime

class Tagger():
    
    def __init__(self):
        pass

    @taticmethod
    def pos(tokens) :
        """Part of speech (POS) tagging identifies the part of speech (noun, verb, 
        adverb, and so on) of each word in a sentence."""
        return [nltk.pos_tag(token) for token in tokens]

    @staticmethod
    def time(tokens):
        """SUTime module is a temporal tagger used to extract temporal expressions
        like dates, times and durations from text."""
        pass

if __name__ == '__main__' :

    """EXAMPLE"""
    
    
