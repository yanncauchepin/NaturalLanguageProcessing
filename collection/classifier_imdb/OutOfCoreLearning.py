import pandas as pd
import numpy as np

"""OUT OF CORE LEARNING"""
"""
Since not everyone has access to supercomputer facilities, we will now apply a
technique called out-of-core learning, which allows us to work with such large
datasets by fitting the classifier incrementally on smaller batches of a dataset.
"""

"""PREPROCESSING DATA"""
import re
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
stop = stopwords.words('english')
def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) \
                  + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

"""
 we define a generator function, stream_docs, that reads in and returns one
 document at a time :
 """
def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv) # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label
"""
To verify that our stream_docs function works correctly, let’s read in the first
document from the movie_data.csv file, which should return a tuple consisting of
the review text as well as the corresponding class label :
"""
next(stream_docs(path='/home/yanncauchepin/PrivateProjects/ArtificialIntelligence/Datasets/NLP/NLP_IMDB/movie_data.csv'))
"""
We will now define a function, get_minibatch, that will take a document stream
from the stream_docs function and return a particular number of documents specified
by the size parameter :
"""
def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y

"""
Unfortunately, we can’t use CountVectorizer for out-of-core learning since it
requires holding the complete vocabulary in memory. Also, TfidfVectorizer needs
to keep all the feature vectors of the training dataset in memory to calculate
the inverse document frequencies. However, another useful vectorizer for text
processing implemented in scikit-learn is HashingVectorizer.
"""
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)
clf = SGDClassifier(loss='log', random_state=1)
doc_stream = stream_docs(path='/home/yanncauchepin/PrivateProjects/ArtificialIntelligence/Datasets/NLP/NLP_IMDB/movie_data.csv')
"""
Using the preceding code, we initialized HashingVectorizer with our tokenizer
function and set the number of features to 2**21. Furthermore, we reinitialized
a logistic regression classifier by setting the loss parameter of SGDClassifier
to 'log'. Note that by choosing a large number of features in HashingVectorizer,
we reduce the chance of causing hash collisions, but we also increase the number
of coefficients in our logistic regression model.
"""
import pyprind
pbar = pyprind.ProgBar(45)
classes = np.array([0, 1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()
"""
Again, we made use of the PyPrind package to estimate the progress of our learning
algorithm. We initialized the progress bar object with 45 iterations and, in the
following for loop, we iterated over 45 mini-batches of documents where each mini-batch
consists of 1,000 documents. Having completed the incremental learning process,
we will use the last 5,000 documents to evaluate the performance of our model.
"""
X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print(f'Accuracy: {clf.score(X_test, y_test):.3f}')
"""
Please note that if you encounter a NoneType error, you may have executed the X_test,
y_test = get_minibatch(...) code twice. Via the previous loop, we have 45 iterations
where we fetch 1,000 documents each. Hence, there are exactly 5,000 documents left
for testing.
If we execute this code twice, then there are not enough documents left in the
generator, and X_test returns None. Hence, if you encounter the NoneType error,
you have to start at the previous stream_docs(...) code again.
"""
"""
As you can see, the accuracy of the model is approximately 87 percent, slightly
below the accuracy that we achieved in the previous section using the grid search
for hyperparameter tuning. However, out-of-core learning is very memory efficient,
and it took less than a minute to complete.
Finally, we can use the last 5,000 documents to update our model :
"""
clf = clf.partial_fit(X_test, y_test)
