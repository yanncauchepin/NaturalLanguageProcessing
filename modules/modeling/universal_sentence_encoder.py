
class USE():
    """The Universal Sentence Encoder (USE) is a model for fetching embeddings at the 
    sentence level. These models are trained using Wikipedia, web news, web question-answer 
    pages, and discussion forums. The pre-trained generalized model can be used for 
    transfer learning directly or can be fine-tuned to a specific task or dataset. 
    The basic building block of USE is an encoder. The USE model can be built using 
    the transformers methodology Capturing Temporal Relationships in Text, or it can 
    be built by combining unigram and bigram representations and feeding them to a 
    neural network to output sentence embeddings through a technique known as deep 
    averaging networks. Several models that have been built using USE-based transfer 
    learning have outperformed state-of-the-art results in the recent past. USE can 
    be used similar to TF-IDF, Word2Vec, Doc2Vec, fastText, and so on for fetching 
    sentence-level embeddings."""

    def __init__(self):
        pass