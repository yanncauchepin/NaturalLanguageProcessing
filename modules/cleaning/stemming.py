import os
import nltk


"""STEMMING"""

"""Word stemming is the process of transforming a word into its root form.
It allows us to map related words to the same stem.
Potential problems arise in the form of over-stemming and under-stemming :
- over-stemming : same root should have been stemmed to different roots
- under-stemming : different roots should have been stemmed to same root
Word stemming can create non-real words."""


from nltk.stem.porter import PorterStemmer
"""The Porter stemming algorithm works only with string."""
def porter_stemming(tokens) :
    porter = PorterStemmer()
    return [porter.stem(token) for token in tokens]


from nltk.stem.snowball import SnowballStemmer
"""The Snowball stemming algorithm (Porter2 or English stemmer) is faster than the
original Porter stemmer. It can work with both string and unicode data."""
def snowball_stemming(tokens) :
    """To test"""
    # snowball stemmer requires the specification of a language parameter
    snowball = SnowballStemmer(language='english')
    return [snwoball(token) for token in tokens]


"""The Lancaster stemming (Paice/Husk stemmer) is faster and more aggressive than
the original Porter stemmer : it produces shorter and more obscur words.
Available throught NLTK."""
def lancaster_stemming(tokens) :
    pass


def dawson_stemming(tokens) :
    pass


def krovetz_stemming(tokens) :
    pass


def lovins_stemming(tokens) :
    pass


"""LEMMATIZATION"""

"""Lemmatization is a process wherein the context is used to convert a word to
its meaningful base form : it aims to obtain the canonical (grammatically correct)
forms of individual words, so-called lemmas. It take into account the neighborhood
context of the word, part-of-speech tags, the meaning of a word, and so on.
Same words can have different lemmas depending on the context."""


nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
"""WordNet is a lexical database of english"""
def word_net_lemmatizer(tokens) :
    """To test"""
    worldnet = WordNetLemmatizer()
    return [worldnet.lemmatize(token) for token in tokens]


def spacy_lemmatizer(tokens) :
    pass


def text_blob_lemmatizer(tokens) :
    pass


def gensim_lemmatizer(tokens) :
    pass


"""STOP WORDS"""

"""Stop words are extremely common words and bear not enough useful information to
distinguish between different classes of documents."""

if not os.path.exists(os.path.join(nltk.data.find('corpora'), 'stopwords', 'english')):
    nltk.download('stopwords')
from nltk.corpus import stopwords
def stop_word_english(tokens) :
    stop = stopwords.words('english')
    return [token for token in tokens if token not in stop]


"""CASE FOLDING"""

"""Case folding is a strategy where all the letters in the are converted to
lowercase. Problems arise in situations where proper nouns are dereived from
common noun terms : case folding will become a bottleneck as case-based
distinction becomes an important feature."""


def case_folding(tokens) :
    """To test"""
    return [token.lower() for token in tokens]


"""N-GRAMS"""

from nltk.util import ngrams
def ngrams_tokenizer(tokens, ngram) :
    """To test"""
    return list(ngrams(tokens,ngram))


"""SORTING"""

def sort(tokens) :
    """To test"""
    return sorted(tokens)


if __name__ == '__main__' :

    """EXAMPLE"""

    '''
    tokens = ["This", "is", "a", "test", "sentence", "with", "some", "stop", "words"]

    stemmed_tokens = porter_stemming_tokenizer(tokens)
    print("Porter stemming output :", stemmed_tokens)

    tokens_without_stopwords = stop_word_english(tokens)
    print("Tokens without stopwords :", tokens_without_stopwords)
    '''
