"""
Topic modeling describes the broad task of assigning topics to unlabeled text documents.
For example, a typical application is the categorization of documents in a large
text corpus of newspaper articles. In applications of topic modeling, we then aim
to assign category labels to those articles, for example, sports, finance, world
news, politics, and local news. Thus, in the context of the broad categories of
machine learning, we can consider topic modeling as a clustering task, a subcategory
of unsupervised learning.
"""

import pandas as pd
import numpy as np

"""LATENT DIRICHLET ALLOCATION (LDA)"""
"""
However, note that while latent Dirichlet allocation is often abbreviated as LDA,
it is not to be confused with linear discriminant analysis, a supervised dimensionality
reduction technique.
LDA is a generative probabilistic model that tries to find groups of words that
appear frequently together across different documents. These frequently appearing
words represent our topics, assuming that each document is a mixture of different
words. The input to an LDA is the bag-of-words model.
Given a bag-of-words matrix as input, LDA decomposes it into two new matrices:
-   A document-to-topic matrix
-   A word-to-topic matrix
LDA decomposes the bag-of-words matrix in such a way that if we multiply those two
matrices together, we will be able to reproduce the input, the bag-of-words matrix,
with the lowest possible error. In practice, we are interested in those topics that
LDA found in the bag-of-words matrix. The only downside may be that we must define
the number of topics beforehand the number of topics is a hyperparameter of LDA
that has to be specified manually.
"""

"""LATENT DIRICHLET ALLOCATION SCIKITLEARN"""
"""
We restrict the analysis to 10 different topics, but readers are encouraged to
experiment with the hyperparameters of the algorithm to further explore the topics
that can be found in this dataset.
"""

"""DATASET IMDB"""
df = pd.read_csv('/home/yanncauchepin/PrivateProjects/ArtificialIntelligence/Datasets/NLP/NLP_IMDB/movie_data.csv', encoding='utf-8')
df = df.rename(columns={"0": "review", "1": "sentiment"})

from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer(stop_words='english',
                        max_df=.1,
                        max_features=5000)
X = count.fit_transform(df['review'].values)

"""
Notice that we set the maximum document frequency of words to be considered to 10
percent (max_df=.1) to exclude words that occur too frequently across documents.
The rationale behind the removal of frequently occurring words is that these might
be common words appearing across all documents that are, therefore, less likely
to be associated with a specific topic category of a given document. Also, we
limited the number of words to be considered to the most frequently occurring
5,000 words (max_features=5000), to limit the dimensionality of this dataset to
improve the inference performed by LDA. However, both max_df=.1 and max_features=5000
are hyperparameter values chosen arbitrarily, and readers are encouraged to tune
them while comparing the results.
The following code example demonstrates how to fit a LatentDirichletAllocation
estimator to the bag-of-words matrix and infer the 10 different topics from the
documents.
Note that the model fitting can take up to 5 minutes or more on a laptop or standard
desktop computer.
"""

from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=10,
                                random_state=123,
                                learning_method='batch')
X_topics = lda.fit_transform(X)

"""
By setting learning_method='batch', we let the lda estimator do its estimation
based on all available training data (the bag-of-words matrix) in one iteration,
which is slower than the alternative 'online' learning method, but can lead to
more accurate results.
The scikit-learn library’s implementation of LDA uses the expectation-maximization
(EM) algorithm to update its parameter estimates iteratively.
"""

"""
After fitting the LDA, we now have access to the components_ attribute of the lda
instance, which stores a matrix containing the word importance (here, 5000) for
each of the 10 topics in increasing order :
"""
lda.components_.shape

"""
To analyze the results, let’s print the five most important words for each of the
10 topics. Note that the word importance values are ranked in increasing order.
Thus, to print the top five words, we need to sort the topic array in reverse order :
"""

n_top_words = 5
feature_names = count.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    print(f'Topic {(topic_idx + 1)}:')
    print(' '.join([feature_names[i]
                    for i in topic.argsort()\
                    [:-n_top_words - 1:-1]]))

"""
Based on reading the five most important words for each topic, you may guess that
the LDA identified the following topics:
1.  Generally bad movies (not really a topic category)
2.  Movies about families
3.  War movies
4.  Art movies
5.  Crime movies
6.  Horror movies
7.  Comedy movie reviews
8.  Movies somehow related to TV shows
9.  Movies based on books
10. Action movies
"""
"""
To confirm that the categories make sense based on the reviews, let’s plot three
movies from the horror movie category (horror movies belong to category 6 at index
position 5) :
"""
horror = X_topics[:, 5].argsort()[::-1]
for iter_idx, movie_idx in enumerate(horror[:3]):
    print(f'\nHorror movie #{(iter_idx + 1)}:')
    print(df['review'][movie_idx][:300], '...')
