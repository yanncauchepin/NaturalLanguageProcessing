nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

class Lemmatizer():
    """Lemmatization is a process wherein the context is used to convert a word to
    its meaningful base form : it aims to obtain the canonical (grammatically correct)
    forms of individual words, so-called lemmas. It take into account the neighborhood
    context of the word, part-of-speech tags, the meaning of a word, and so on.
    Same words can have different lemmas depending on the context."""
    
    def __init__(self):
        pass

    @staticmethod
    def word_net(tokens) :
        """WordNet is a lexical database of english"""
        worldnet = WordNetLemmatizer()
        return [worldnet.lemmatize(token) for token in tokens]
    
    @staticmethod
    def spacy(tokens) :
        pass
    
    @staticmethod
    def text_blob(tokens) :
        pass
    
    @staticmethod
    def gensim(tokens) :
        pass