# ------ Search ------

'''
Use cosine similarity => cosine angle between the doc vector and the prompt vector.
'''
import re
from collections import defaultdict
from numpy import dot
from numpy.linalg import norm

from config import confidenceThresholdIntent, confidenceThresholdIntentconfirm
from utils import confirmation
from nlg import generateContextualError, addDiscourseMarker

# Mapping to translate intent codes into human-readable descriptions
intentDescriptions = {
    "small": "small talk",
    "question": "questions about books",
    "identity": "identity management",
    "discover": "discovering features",
    "order": "ordering books",
    "location": "store locations",
    "check": "checking orders",
    "thank": "thanking",
    "opening": "opening hours"
}

def getCosineSimilarity(query_doc, doc):
    # norm() gives the l2-norm of a vector:
    #   - square root of, the sum of, the squares of a vector's components.
    # dot() gives the dot-product of two vectors.
    query_norm = norm(query_doc)
    doc_norm = norm(doc)
    # Check for zero vectors to avoid division by zero
    if doc_norm == 0 or query_norm == 0:
        return 0.0
    return dot(query_doc, doc) / (norm(query_doc) * norm(doc))

def searchIntent(inverted_index, query, vectoriser, tfidf, X_train_tf, intents):
    qCounts = vectoriser.transform([query])
    qTfidf = tfidf.transform(qCounts)

    # Attempt to use the inverted index to compute dot-products efficiently.
    # inverted_index may be either the saved index object (with 'index' and 'doc_norms')
    # or the raw inverted index dict (backward-compat). Handle both.
    if isinstance(inverted_index, dict) and 'index' in inverted_index and 'doc_norms' in inverted_index:
        index = inverted_index['index']
        doc_norms = inverted_index['doc_norms']
    else:
        # Backwards compatibility: treat the passed value as the index and compute norms from X_train_tf
        index = inverted_index
        try:
            dVecs = X_train_tf.toarray()
            doc_norms = [float(norm(dVec)) for dVec in dVecs]
        except Exception:
            sq = X_train_tf.multiply(X_train_tf).sum(axis=1)
            doc_norms = [float(sq[i, 0]) ** 0.5 for i in range(sq.shape[0])]

    # Now compute dot-products only for documents that share at least one query term.
    q_coo = qTfidf.tocoo()
    # precompute query norm
    qVec = qTfidf.toarray().flatten()
    q_norm = norm(qVec)

    # accumulators for docId -> dot(query, doc)
    accum = defaultdict(float)
    feature_names = vectoriser.get_feature_names_out()
    for termId, q_weight in zip(q_coo.col, q_coo.data):
        term = feature_names[termId]
        postings = index.get(term)
        if not postings:
            continue
        for docId, d_weight in postings.items():
            accum[docId] += q_weight * d_weight

    similarity = []
    # If accum is empty (no overlapping terms) fall back to dense cosine across all docs
    if not accum:
        # dense fallback (same as before)
        dVecs = X_train_tf.toarray()
        for docId in range(len(intents)):
            dVec = dVecs[docId]
            similarity.append((docId, getCosineSimilarity(qVec, dVec)))
    else:
        # compute cosine similarity for docs in accumulator
        for docId, dotprod in accum.items():
            denom = q_norm * (doc_norms[docId] if docId < len(doc_norms) else 0)
            score = dotprod / denom if denom != 0 else 0.0
            similarity.append((docId, score))

    # Return the most likely intent
    similarity.sort(key=lambda x: x[1], reverse=True)
    if not similarity:
        return None
    if similarity[0][1] > confidenceThresholdIntent:
        return similarity[0]
    elif similarity[0][1] > confidenceThresholdIntentconfirm:
        intentCode = intents[similarity[0][0]][1]
        intentDesc = intentDescriptions.get(intentCode, intentCode)
        print(f"To confirm, you're asking about {intentDesc}?")
        if confirmation():
            # If the user confirms, add their prompt + confirmed intent to the intents.csv dataset
            addIntentExample(query,intents[similarity[0][0]][1])
            return similarity[0]
        else:
            error = generateContextualError('generic')
            print(addDiscourseMarker('clarification', f"{error} Try asking me 'what can you do' for specific examples."))

'''
Helper function for adding intent examples to the dataset.
'''
def addIntentExample(prompt, intent):
    with open('intents.csv', 'a', encoding='utf-8') as f:
        f.write(f'"{prompt}","{intent}"\n')

def question(qa, question_text, vectoriser, tfidf, X_train_tf, inv_index=None):
    """
    Search QA using cosine similarity. If an inverted-index object (`inv_index`) is
    provided (the structure returned by `generateInvertedIndex`), use the postings
    to compute dot-products sparsely and then divide by stored norms. Otherwise
    fall back to the dense approach.
    """
    # Vectorise the query
    qCounts = vectoriser.transform([question_text])
    qTfidf = tfidf.transform(qCounts)

    # If an inverted index object is provided, use it for sparse scoring
    if inv_index is not None and isinstance(inv_index, dict) and 'index' in inv_index and 'doc_norms' in inv_index:
        index = inv_index['index']
        doc_norms = inv_index['doc_norms']

        q_coo = qTfidf.tocoo()
        qVec = qTfidf.toarray().flatten()
        q_norm = norm(qVec)

        accum = defaultdict(float)
        feature_names = vectoriser.get_feature_names_out()
        for termId, q_weight in zip(q_coo.col, q_coo.data):
            term = feature_names[termId]
            postings = index.get(term)
            if not postings:
                continue
            for docId, d_weight in postings.items():
                accum[docId] += q_weight * d_weight

        similarity = []
        if not accum:
            # fallback to dense
            dVecs = X_train_tf.toarray()
            qVec = qTfidf.toarray().flatten()
            for docId in range(len(qa)):
                dVec = dVecs[docId]
                similarity.append((docId, getCosineSimilarity(qVec, dVec)))
        else:
            for docId, dotprod in accum.items():
                denom = q_norm * (doc_norms[docId] if docId < len(doc_norms) else 0)
                score = dotprod / denom if denom != 0 else 0.0
                similarity.append((docId, score))

    else:
        # No inverted index supplied: dense scoring over all QA docs
        qVec = qTfidf.toarray().flatten()
        dVecs = X_train_tf.toarray()
        similarity = []
        for docId in range(len(qa)):
            dVec = dVecs[docId]
            similarity.append((docId, getCosineSimilarity(qVec, dVec)))

    # Return the most likely answer
    similarity.sort(key=lambda x: x[1], reverse=True)
    if not similarity:
        error = generateContextualError('generic')
        return addDiscourseMarker('clarification', f"{error} I don't have enough information to answer that question.")
    if qa[similarity[0][0]][1] == "Answer":
        error = generateContextualError('generic')
        return addDiscourseMarker('clarification', f"{error} Try rephrasing your question or asking something else.")
    return qa[similarity[0][0]][1]
