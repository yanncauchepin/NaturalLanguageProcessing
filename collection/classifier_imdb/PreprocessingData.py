"""
Sentiment analysis, sometimes also called opinion mining, is a popular subdiscipline
of the broader field of NLP; it is concerned with analyzing the sentiment of
documents. A popular task in sentiment analysis is the classification of documents
based on the expressed opinions or emotions of the authors with regard to a
particular topic.

 The movie review dataset consists of 50,000 polar movie reviews that are labeled
 as either positive or negative; here, positive means that a movie was rated with
 more than six stars on IMDb, and negative means that a movie was rated with fewer
 than five stars on IMDb.
"""


import pyprind
import pandas as pd
import os
import sys
import numpy as np

"""DATASET IMDB"""

basepath = '/home/yanncauchepin/PrivateProjects/ArtificialIntelligence/Datasets/NLP/NLP_IMDB'

labels = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(50000, stream=sys.stdout)
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path, file),
                      'r', encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]],
                           ignore_index=True)
            pbar.update()
df.columns = ['review', 'sentiment']
"""
We first initialized a new progress bar object, pbar, with 50,000 iterations,
which was the number of documents we were going to read in. Using the nested for
loops, we iterated over the train and test subdirectories in the main Imdb
directory and read the individual text files from the pos and neg subdirectories
that we eventually appended to the df pandas DataFrame, together with an integer
class label (1 = positive and 0 = negative).
Since the class labels in the assembled dataset are sorted, we will now shuffle
the DataFrame using the permutation function from the np.random submodule—this
will be useful for splitting the dataset into training and test datasets in later
sections, when we will stream the data from our local drive directly.
For our own convenience, we will also store the assembled and shuffled movie review
dataset as a CSV file.
"""
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('movie_data.csv', index=False, encoding='utf-8')

df = pd.read_csv('movie_data.csv', encoding='utf-8')
# the following column renaming is necessary on some computers:
df = df.rename(columns={"0": "review", "1": "sentiment"})
df.head(3)
df.shape

"""CLEAN TEXT DATA"""

"""
Let’s display the last 50 characters from the first document in the reshuffled
movie review dataset.
"""
df.loc[0, 'review'][-50:]
"""
As you can see here, the text contains HTML markup as well as punctuation and
other non-letter characters. While HTML markup does not contain many useful
semantics, punctuation marks can represent useful, additional information in
certain NLP contexts. However, for simplicity, we will now remove all punctuation
marks except for emoticon characters, such as :), since those are certainly useful
for sentiment analysis.
"""

"""REGEX"""
"""
To accomplish this task, we will use Python’s regular expression (regex) library,
re.
"""
import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text
"""
Via the first regex, <[^>]*>, we tried to remove all of the HTML markup from the
movie reviews. Although many programmers generally advise against the use of regex
to parse HTML, this regex should be sufficient to clean this particular dataset.
Since we are only interested in removing HTML markup and do not plan to use the
HTML markup further, using regex to do the job should be acceptable. However, if
you prefer to use sophisticated tools for removing HTML markup from text, you can
take a look at Python’s HTML parser module.
After we removed the HTML markup, we used a slightly more complex regex to find
emoticons, which we temporarily stored as emoticons. Next, we removed all non-word
characters from the text via the regex [\W]+ and converted the text into lowercase
characters.
Dealing with word capitalization. In the context of this analysis, we assume that
the capitalization of a word—for example, whether it appears at the beginning of
a sentence—does not contain semantically relevant information. However, note that
there are exceptions; for instance, we remove the notation of proper names. But
again, in the context of this analysis, it is a simplifying assumption that the
letter case does not contain information that is relevant for sentiment analysis.
Eventually, we added the temporarily stored emoticons to the end of the processed
document string. Additionally, we removed the nose character (- in :-)) from the
emoticons for consistency.
Although the addition of the emoticon characters to the end of the cleaned document
strings may not look like the most elegant approach, we must note that the order
of the words doesn’t matter in our bag-of-words model if our vocabulary consists
of only one-word tokens.
"""
preprocessor(df.loc[0, 'review'][-50:])
preprocessor("</a>This :) is :( a test :-)!")

df['review'] = df['review'].apply(preprocessor)

"""TOKEN"""
"""
After successfully preparing the movie review dataset, we now need to think about
how to split the text corpora into individual elements. One way to tokenize
documents is to split them into individual words by splitting the cleaned documents
at their whitespace characters.
"""
def tokenizer(text):
    return text.split()
tokenizer('runners like running and thus they run')

"""World Stemming"""
"""
In the context of tokenization, another useful technique is word stemming, which
is the process of transforming a word into its root form. It allows us to map
related words to the same stem.
The Natural Language Toolkit for Python implements the Porter stemming algorithm,
which we will use in the following code section.
"""
import nltk

from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
tokenizer_porter('runners like running and thus they run')

"""Stemming algorithms"""
"""
The Porter stemming algorithm is probably the oldest and simplest stemming algorithm.
Other popular stemming algorithms include the newer Snowball stemmer (Porter2 or
English stemmer) and the Lancaster stemmer (Paice/Husk stemmer). While both the
Snowball and Lancaster stemmers are faster than the original Porter stemmer, the
Lancaster stemmer is also notorious for being more aggressive than the Porter
stemmer, which means that it will produce shorter and more obscure words. These
alternative stemming algorithms are also available through the NLTK package.
While stemming can create non-real words, such as 'thu' (from 'thus'), as shown
in the previous example, a technique called lemmatization aims to obtain the
canonical (grammatically correct) forms of individual words—the so-called lemmas.
However, lemmatization is computationally more difficult and expensive compared
to stemming and, in practice, it has been observed that stemming and lemmatization
have little impact on the performance of text classification.
"""

"""Stop Word Removal"""
"""
Stop words are simply those words that are extremely common in all sorts of texts
and probably bear no (or only a little) useful information that can be used to
distinguish between different classes of documents. Examples of stop words are
is, and, has, and like. Removing stop words can be useful if we are working with
raw or normalized term frequencies rather than tf-idfs, which already downweight
the frequently occurring words.
To remove stop words from the movie reviews, we will use the set of 127 English
stop words that is available from the NLTK library, which can be obtained by calling
the nltk.download function.
"""
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes'
    ' running and runs a lot')
    if w not in stop]
