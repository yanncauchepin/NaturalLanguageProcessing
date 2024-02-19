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


if __name__ == '__main__' :

    """EXAMPLE"""
