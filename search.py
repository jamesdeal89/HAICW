# ------ Search ------

'''
Use cosine similarity => cosine angle between the doc vector and the prompt vector.
'''
from collections import defaultdict
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

def searchIntent(inverted_index: dict, query: str, vectoriser, tfidf, intents: list, skip_confirmation: bool = False) -> tuple[str | None, float]:
    qCounts = vectoriser.transform([query])
    qTfidf = tfidf.transform(qCounts)

    index = inverted_index['index']
    doc_norms = inverted_index['doc_norms']

    # Compute dot-products only for documents that share at least one query term.
    q_coo = qTfidf.tocoo()
    qVec = qTfidf.toarray().flatten()
    q_norm = norm(qVec)

    # Accumulators for docId -> dot(query, doc)
    accum = defaultdict(float)
    feature_names = vectoriser.get_feature_names_out()
    for termId, q_weight in zip(q_coo.col, q_coo.data):
        term = feature_names[termId]
        postings = index.get(term)
        if not postings:
            continue
        for docId, d_weight in postings.items():
            accum[docId] += q_weight * d_weight

    # Compute cosine similarity for docs in accumulator
    similarity = []
    for docId, dotprod in accum.items():
        denom = q_norm * doc_norms[docId]
        score = dotprod / denom if denom != 0 else 0.0
        similarity.append((docId, score))

    # Return the most likely intent
    similarity.sort(key=lambda x: x[1], reverse=True)
    if not similarity:
        return None
    if similarity[0][1] > confidenceThresholdIntent:
        return similarity[0]
    elif similarity[0][1] > confidenceThresholdIntentconfirm:
        # If skip_confirmation is True (e.g., during evaluation), accept the intent without prompting
        if skip_confirmation:
            return similarity[0]
        
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
            return None

'''
Helper function for adding intent examples to the dataset.
'''
def addIntentExample(prompt: str, intent: str) -> None:
    with open('intents.csv', 'a', encoding='utf-8') as f:
        f.write(f'"{prompt}","{intent}"\n')

def question(qa: list, question_text: str, vectoriser, tfidf, inv_index: dict) -> str:
    """
    Search QA using cosine similarity with the inverted index.
    Uses sparse dot-products via postings and precomputed document norms.
    """
    # Vectorise the query
    qCounts = vectoriser.transform([question_text])
    qTfidf = tfidf.transform(qCounts)

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

    # Compute cosine similarity for docs in accumulator
    similarity = []
    for docId, dotprod in accum.items():
        denom = q_norm * doc_norms[docId]
        score = dotprod / denom if denom != 0 else 0.0
        similarity.append((docId, score))

    # Return the most likely answer
    similarity.sort(key=lambda x: x[1], reverse=True)
    if not similarity:
        error = generateContextualError('generic')
        return f"{error} I don't have enough information to answer that question."
    if qa[similarity[0][0]][1] == "Answer":
        error = generateContextualError('generic')
        return f"{error} Try rephrasing your question or asking something else."
    return qa[similarity[0][0]][1]

def bookDescSearch(bookDesc: list, userPrompt: str, vectoriser, tfidf, invIndex: dict) -> list[tuple[int, float]]:
    """
    Search book descriptions using cosine similarity with the inverted index.
    Uses sparse dot-products via postings and precomputed document norms.
    """
    # Vectorise the query
    qCounts = vectoriser.transform([userPrompt])
    qTfidf = tfidf.transform(qCounts)

    index = invIndex['index']
    doc_norms = invIndex['doc_norms']

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

    # Compute cosine similarity for docs in accumulator
    similarity = []
    for docId, dotprod in accum.items():
        denom = q_norm * doc_norms[docId]
        score = dotprod / denom if denom != 0 else 0.0
        similarity.append((docId, score))

    # Return the most likely book matches (top 2 for fallback)
    similarity.sort(key=lambda x: x[1], reverse=True)
    if not similarity:
        return None
    # Return top 2 results if available for fallback handling
    if len(similarity) >= 2:
        return [(similarity[0][0], similarity[0][1]), (similarity[1][0], similarity[1][1])]
    return [(similarity[0][0], similarity[0][1])]
