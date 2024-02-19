import nltk


def _tokenizer(text) :
    return text.split()


def tokenizer(text) :
    """To test"""
    return nltk.tokenize(text)


from nltk.tokenize import RegexpTokenizer
def customized_regex_tokenizer(regex, text) :
    """To test"""
    regexp = RegexpTokenizer(regex)
    return regexp.tokenize(text)


def blank_line_tokenizer(text) :
    """nltk BlankLine"""
    pass


def word_punct_tokenizer(text) :
    """nltk WordPunct"""
    pass


from nltk.tokenize import TreebankWordTokenizer
def treebank_word_tokenizer(text) :
    """To test"""
    treebankword = TreebankWordTokenizer()
    return treebankword.tokenize(text)


from nltk.tokenize import TweetTokenizer
def tweet_tokenizer(text) :
    """To test"""
    # can add parameters strip_handles=True, reduce_len=True, preserve_case=False
    tweet = TweetTokenizer()
    return tweet.tokenize(text)


if __name__ == '__main__' :

    """EXAMPLE"""

    '''
    text = "This is a test sentence for tokenization."

    tokens = split_tokenizer(text)
    print("Tokenization output :", tokens)

    takens = nltk_tokenizer(text)
    print("Tokenization output :", tokens)
    '''
