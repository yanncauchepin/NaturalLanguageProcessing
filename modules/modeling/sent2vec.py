
class Sent2Vec():
    """Sent2Vec combines the continuous bag-of-words approach we discussed regarding 
    Word2Vec, along with the fastText thought process of using constituent n-gram, 
    to build sentence embeddings. Contextual word embeddings and target word embeddings 
    were learned by trying to predict the target words based on the context of the words, 
    similar to the C-BOW approach. However, they extended the C-BOW methodology to define 
    sentence embeddings as the average of the context word embeddings present in the sentence, 
    wherein context word embeddings are not restricted to unigrams but extended to n-grams 
    in each sentence, similar to the fastText approach. The sentence embedding would then 
    be represented as the average of these n-gram embeddings. Research has shown that 
    Sent2Vec outperforms Doc2Vec in the majority of the tasks it undertakes and that 
    it is a better representation method for sentences or documents. The Sent2Vec 
    library is an open sourced implementation of the model that's built on top of 
    fastText and can be used similar to the Doc2Vec and fastText models, which we have 
    discussed extensively so far."""
    
    def __init__(self):
        pass