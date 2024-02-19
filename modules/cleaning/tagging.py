import nltk


nltk.download('tagsets')
# nltk.help.upenn_tagset()
#nltk.download('averaged_perceptron_tagger')
"""Part of speech (POS) tagging identifies the part of speech (noun, verb, adverb,
and so on) of each word in a sentence."""
def pos_tag(tokens) :
    """To test"""
    return [nltk.pos_tag(token) for token in tokens]


from sutime import SUTime
"""SUTime module is a temporal tagger used to extract temporal expressions
like dates, times and durations from text."""


if __name__ == '__main__' :

    """EXAMPLE"""
