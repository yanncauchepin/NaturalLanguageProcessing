"""encapsulating each word as a combination of character n-grams. Each of these 
n-grams has a vector representation. Word representations are actually a result 
of the summation of their character n-grams"""
"""two- and three-character n-grams for the word language:
    la, lan, an, ang, ng, ngu, ..., ag, age, ge"""
    
"""fastText leads to parameter sharing among various words that have any overlapping 
n-grams. We capture their morphological information from sub-words to build an 
embedding for the word itself. Also, when certain words are missing from the 
training vocabulary or rarely occur, we can still have a representation for them 
if their n-grams are present as part of other words."""


from gensim.models import FastText
from gensim.test.utils import common_texts

model = FastText(vector_size=5, window=3, min_count=1)
model.build_vocab(corpus_iterable=common_texts)
model.train(corpus_iterable=common_texts, total_examples=len(common_texts), epochs=10)
model.wv.key_to_index
"vector of the word human"
model.wv["human"]

"""vec(computer) + vec(interface) - vec(human)"""
model.wv.most_similar(positive=['computer', 'interface'], negative=['human'])

"""word representations in FastText are built using the n-grams, min_n, and max_n 
characters, this helps us by setting the minimum and maximum lengths of the character 
n-grams so that we can build representations."""
model = FastText(vector_size=5, window=3, min_count=1, min_n=1, max_n=5)
model.build_vocab(corpus_iterable=common_texts)
model.train(corpus_iterable=common_texts, total_examples=len(common_texts), epochs=10)
"""try and extend our model so that it incorporates new sentences and vocabulary. """
sentences_to_be_added = [["I", "am", "learning", "Natural", "Language", "Processing"],
 ["Natural", "Language", "Processing", "is", "cool"]]
model.build_vocab(sentences_to_be_added, update=True)
model.train(corpus_iterable=common_texts, total_examples=len(sentences_to_be_added), epochs=10)

"""The original fastText research paper extended on the Skip-gram approach for Word2Vec, 
but today, both the Skip-gram and continuous bag-of-words approach can be used. 
Pre-trained fastText models across multiple languages are available online and can 
be directly used or fine-tuned so that we can understand a specific dataset better.
fastText can be applied to solve a plethora of problems such as spelling correction, 
auto suggestions, and so on since it is based on sub-word representation. Datasets 
such as user search query, chatbots or conversations, reviews, and ratings can be used 
to build fastText models. """

if __name__ == '__main__' :
    pass