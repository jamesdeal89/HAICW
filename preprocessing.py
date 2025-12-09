# ====== PREPROCESSING ======
import os
import csv
import pickle
import time
import json
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem.snowball import PorterStemmer
from nltk.corpus import stopwords
from nltk import download
import re
from numpy.linalg import norm

download('stopwords', quiet=True)

# Cache timeout: 1 day in seconds
CACHE_TIMEOUT = 24 * 60 * 60

"""
Remove pickle cache files if they are older than 1 day.
This ensures the cache regenerates periodically with fresh data.
"""
def clearOldPickles() -> None:
    pickle_files = [
        'intents.pickle',
        'qa.pickle',
        'bookDesc.pickle',
        'XtrainTfIntents.pickle',
        'countIntents.pickle',
        'tfidfIntents.pickle',
        'invIdx.pickle',
        'XtrainTfQa.pickle',
        'countQa.pickle',
        'tfidfQa.pickle',
        'invIdxQa.pickle',
        'XtrainTfBookDesc.pickle',
        'countBookDesc.pickle',
        'tfidfBookDesc.pickle',
        'invIdxBookDesc.pickle'
    ]
    
    current_time = time.time()
    for pickle_file in pickle_files:
        if os.path.exists(pickle_file):
            file_age = current_time - os.path.getmtime(pickle_file)
            if file_age > CACHE_TIMEOUT:
                os.remove(pickle_file)
                print(f"Cleared old cache file: {pickle_file}")

'''
Read in the CSV of example prompts labelled with respetive intents.
As similarity based intent matching, resolve the user prompt to the 
most similar entry in the corpus of prompt docs, resolve to respective intent label.
'''
def readIntentsCsv() -> list[list[str]]:
    # If we have the intents object saved to disk, just load and return that
    if os.path.exists('intents.pickle'):
        with open("intents.pickle", "rb") as f:
            return pickle.load(f)

    # If not saved to disk, re-load from csv
    intents = []
    with open('intents.csv', 'r', encoding='utf-8', newline='') as f:
        r = csv.reader(f, delimiter=',')
        for row in r:
            # format: ["prompt","intent"]
            intents.append(row)
    
    # Save to disk for next time
    with open(f"intents.pickle", "wb") as f:
        pickle.dump(intents, f)

    return intents

# ------ Stemming, Vectorising, Weighting ------

p_stemmer = PorterStemmer()
def stemmed_words(doc: str) -> str:
    tokens = re.findall(r'\b\w+\b', doc.lower())
    return [p_stemmer.stem(token) for token in tokens]
def stemmed_filterStopwords_words(doc: str) -> str:
    tokens = re.findall(r'\b\w+\b', doc.lower())
    return [p_stemmer.stem(token) for token in tokens if token not in stopwords.words('english')]

def stemVectorWeight(intents: list, filterStopwords: bool, pName: str, pName1: str, pName2: str, textIndex: int = 0):
    # If already saved to disk, simply load and return the disk objects
    if os.path.exists(pName) and os.path.exists(pName1) and os.path.exists(pName2):
        with open(pName, "rb") as f:
            XtrainTf = pickle.load(f)
        with open(pName1, "rb") as f:
            count = pickle.load(f)
        with open(pName2, "rb") as f:
            tfidf = pickle.load(f)
        return (XtrainTf, count, tfidf)
    # If not found on disk, re-generate.

    # Get just the prompts to vectorise, but maintain indexing to resolve to labels.
    prompts = []
    for pair in intents: prompts.append(pair[textIndex])

    # Initialise and run the count based vectoriser on the prompts, 
    # also using the stemmer.
    if filterStopwords:
        count_vect = CountVectorizer(tokenizer=stemmed_filterStopwords_words, lowercase=True)
    else:
        count_vect = CountVectorizer(tokenizer=stemmed_words, lowercase=True)
    X_train_counts = count_vect.fit_transform(prompts)

    # Term weighting: Term frequency - Inverse document frequency
    tf_transformer = TfidfTransformer(use_idf=True, sublinear_tf=True, norm=None).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    # Save to disk for next run
    with open(pName, "wb") as f:
        pickle.dump(X_train_tf, f)
    with open(pName1, "wb") as f:
        pickle.dump(count_vect, f)
    with open(pName2, "wb") as f:
        pickle.dump(tf_transformer, f)
    return (X_train_tf, count_vect, tf_transformer)

# ------ Term Document Matrix ------

'''
Building an *Inverted Index*:
Term-document (TD) matrix is imply the tranpose of the document-term matrix.
- Instead of which terms appear in a document:
    - care about which documents contain a specific term.
Advantage:
- Do *not* need to build the full matrix.
- Just store list of documents containing a term as a list.
'''

def createFloatDict() -> type:
    return defaultdict(float)

def generateInvertedIndex(count_vect, X_train_tf, pName: str) -> dict:
    # If a saved inverted index exists on disk, load and return it.
    if os.path.exists(pName):
        with open(pName, "rb") as f:
            return pickle.load(f)

    # Format: {term: {docId: tfidfScore}}
    inverted_index = defaultdict(createFloatDict)

    feature_names = count_vect.get_feature_names_out()
    # Convert to co-ordinate format to make iteration more efficient.
    tfid_matrix = X_train_tf.tocoo()
    for docId, termId, score in zip(tfid_matrix.row, tfid_matrix.col, tfid_matrix.data):
        term = feature_names[termId]
        inverted_index[term][docId] = score

    # Precompute per-document l2 norms so cosine similarity can be computed quickly
    try:
        dVecs = X_train_tf.toarray()
        # list of floats: norm of each document vector
        doc_norms = [float(norm(dVec)) for dVec in dVecs]
    except Exception:
        # fallback: compute norms from sparse representation
        sq = X_train_tf.multiply(X_train_tf).sum(axis=1)
        # sq is a matrix; convert each to float and sqrt
        doc_norms = [float(sq[i, 0]) ** 0.5 for i in range(sq.shape[0])]

    index_obj = {
        'index': inverted_index,
        'doc_norms': doc_norms
    }

    # Save to disk for next time (store both index and doc norms)
    with open(pName, "wb") as f:
        pickle.dump(index_obj, f)
    return index_obj

# 'Postings' => list of docs that contain each term, alongside the term's importance in those docs.
# inverted_index resolves a term in the vocab to it's respective postings.

'''
Load QA pairs.
Format in the same way as intents to enable re-use of similarity based matching functions.
'''
def readQaCsv() -> list[list[str]]:
    # If we have the QA object saved to disk, just load and return that
    if os.path.exists('qa.pickle'):
        with open("qa.pickle", "rb") as f:
            return pickle.load(f)

    # If not saved to disk, re-load from csv
    qa = []
    with open('qa.csv', 'r', encoding='utf-8', newline='') as f:
        r = csv.reader(f, delimiter=',')
        for row in r:
            # format: ["question","intent"]
            qa.append(row)
    
    # Save to disk for next time
    with open(f"qa.pickle", "wb") as f:
        pickle.dump(qa, f)

    return qa

'''
Load book descriptions.
Format in the same way as intents to enable re-use of similarity based matching functions.
'''
def readBookDescriptions() -> list[list[str]]:
    # If we have the book descriptions saved to disk, just load and return that
    if os.path.exists('bookDesc.pickle'):
        with open("bookDesc.pickle", "rb") as f:
            return pickle.load(f)

    # If not saved to disk, re-load from JSON
    bookDesc = []
    with open('stock.json', 'r', encoding='utf-8') as f:
        stock = json.load(f)
    
    for book in stock['stock']:
        # format: [description, name] - description first for TF-IDF vectorization
        bookDesc.append([book.get('description', ''), book['name']])
    
    # Save to disk for next time
    with open(f"bookDesc.pickle", "wb") as f:
        pickle.dump(bookDesc, f)

    return bookDesc
