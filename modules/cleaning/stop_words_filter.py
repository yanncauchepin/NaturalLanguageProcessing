import os
import nltk
if not os.path.exists(os.path.join(nltk.data.find('corpora'), 'stopwords', 'english')):
    nltk.download('stopwords')
from nltk.corpus import stopwords

class StopWordsFilter():
    """Stop words are extremely common words and bear not enough useful information 
    to distinguish between different classes of documents."""
    
    def __init__(self):
        pass

    @staticmethod
    def english(tokens) :
        stop = stopwords.words('english')
        return [token for token in tokens if token not in stop]
    
if __name__ == "__main__":
    
    """EXAMPLE"""

    tokens = ["This", "is", "a", "test", "sentence", "with", "some", "stop", "words"]
    
    tokens_without_stopwords = StopWordsFilter.english(tokens)
    print("Tokens without stopwords :", tokens_without_stopwords)
