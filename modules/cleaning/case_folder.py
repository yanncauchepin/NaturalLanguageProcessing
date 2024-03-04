class CaseFolder():
    """Case folding is a strategy where all the letters in the are converted to
    lowercase. Problems arise in situations where proper nouns are dereived from
    common noun terms : case folding will become a bottleneck as case-based
    distinction becomes an important feature."""

    def __init__(self):
        pass

    @staticmethod
    def lower(tokens) :
        return [token.lower() for token in tokens]
    
if __name__ == '__main__':
    
    """EXAMPLE"""
    
    tokens = ["Hello", "World", "This", "is", "An", "Example"]
    
    folded_tokens = CaseFolder.lower(tokens)
    print(folded_tokens)
    