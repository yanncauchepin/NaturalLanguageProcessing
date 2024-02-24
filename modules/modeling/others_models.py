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
