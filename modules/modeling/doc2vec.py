"""Doc2vec is a unsupervised algorithm.
There is two ways of building paragraph vectors:
- Distributed Memory Model of Paragraph Vectors (PV-DM) where paragraph vectors
are concatenated with the word vectors.
- Distributed Bag-of-words of Paragraph Vectors (PV-DBOW) where word vectors
are not taken into account. Analogous to Skip-gram approach in Word2vec.
It is simpler and more memory-efficient as word vectors do not need to be
stored.
These learned representations can serve to various tasks such as the
classification/clustering of the document."""
def doc2vec() :
    pass


from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
"""common_texts is a small document corpus.
Doc2Vec expects sentences in TaggedDocument format.
Doc2Vec expects a list of tokens as input for each document."""
"""Transform tokenized document into TaggedDocument format"""
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
documents

"""Build and train a basic Doc2Vec model.
Vector size of 5 denotes that each document will be represented by a vector of
five floating-point values."""
model = Doc2Vec(documents, vector_size=5, min_count=1, workers=4, epochs = 40)
model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
"""Validate the vector size for the document embeddings.
Check whether the number of document vectors being built is equal to the number
of documents being used in the training process.
Checking the length of our vocabulary."""
model.vector_size
len(model.dv)
len(model.wv.key_to_index)
model.wv.key_to_index
"""let's build a document vector for a new sentence."""
vector = model.infer_vector(['user', 'interface', 'for', 'computer'])
print(vector)

"""min_count"""
model = Doc2Vec(documents, vector_size=50, min_count=3, epochs=40)
model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
len(model.wv.key_to_index)
model.wv.key_to_index
vector = model.infer_vector(['user', 'interface', 'for', 'computer'])
"""paragraph vector.
vector size is 50 and only 4 terms are in the vocabulary. 
because min_count was  3 and, consequently, terms that were equal to or greater 
than 3 terms are present in the vocabulary now."""
print(vector)

"""dm parameter for switching between PVDM and PVDBOW
dm = 0 for PVDBOW, dm = 1 for PVDM"""
model = Doc2Vec(documents, vector_size=50, min_count=2, epochs=40, dm=1)
model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

"""PVDM takes word vectors into account + two additional parameters :
dm_concat : 1 to concatenate context vectors while trying to predict the target word
    leads to building a larger model since multiple word embeddings get concatenated.
    set to 1 to not concatenate and use lighter model
dm_mean : to sum or average the context vectors instead of concatenating them
    set to 1 to take the mean of the context word vectors
    set to 0 to take the sum of the context word vectors"""

model = Doc2Vec(documents, vector_size=50, min_count=2, epochs=40, window=2, dm=1, 
                alpha=0.3, min_alpha=0.05, dm_concat=1)
model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
model = Doc2Vec(documents, vector_size=50, min_count=2, epochs=40, window=2, dm=1, 
                dm_concat=0, dm_mean=1, alpha=0.3, min_alpha=0.05)
model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

"""window size controls distance between the word under concentration and the word to be
predicted"""
model = Doc2Vec(documents, vector_size=50, min_count=2, epochs=40, window=2, dm=0)
model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

"""learning rate can be specifiy (initial) with alpha. min alpha specify the value
the learning rate should dorp to over the course of training."""
model = Doc2Vec(documents, vector_size=50, min_count=2, epochs=40, window=2, dm=1, 
                alpha=0.3, min_alpha=0.05)
model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

"""other parameters:
    negative : enabling negative sampling similar to word2vec
    max_vocab_size : limit the vocabulary"""
    

if __name__ == '__main__' :

    """EXAMPLE"""
