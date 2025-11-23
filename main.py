
# ====== COMP3074 Coursework Chatbot ======
# Author: James Deal
# ID: 20551937

'''
NOTE: Allowed libraries include: 
- Python standard library.
- numpy,
- scipy,
- scikit-learn (and it's dependencies),
- nltk (EXCEPT nltk.chat)
'''

'''
CHECKLIST OF EXPECTED FEATURES:
1. Intent matching. [x]
2. Identity management. [x]
3. Transactions. [ ]
4. Information retrieval & Question answering. [x]
5. Small talk. [x]
'''

'''
IDEA:
Book ordering system.
Transaction: ordering books, scheduling pickup/delivery.
Q&A: book recomendations, location opening times.
'''

# TODO: recomendations based on cosine similarity to book description

'''
FOR THE CHECKPOINT:
Requirements:
Implement *similarity-based* intent matching, 
which routes the flow of the chatbot to functions for handling:
- Small talk,
- Discoverability (what can the bot do?),
- Identity management,
- Question answering.

They will test with:
Hi or Hello [greet the user and ask their name]
What is my name [tell the user their name]
How are you [answer]
What can you do [answer]
What are stocks and bonds [answer from QA dataset]

NOTE: For purposes of checkpoint, need to use *stocks QA dataset* provided, 
for final submission can change.

CHECKPOINT WAS COMPLETED AND BONUS MARKS RECEIVED.
'''

'''
NOTE on Q&A dataset:
Static dataset was partially genAI generated to bulk out examples. 
NONE of the code, processing, etc are AI generated.
'''

confidenceThresholdQA = 0.6
confidenceThresholdQAconfirm = 0.4
confidenceThresholdIntent = 0.5
confidenceThresholdIntentconfirm = 0.3

def main():
    print("======== STARTING BOOKSTORE CHATBOT ========")
    # NOTE: below ASCII art was generated via https://patorjk.com/software/taag/
    print("""
    ██████  ██       █████   ██████ ██   ██ ███████ ███    ███ ██ ████████ ██   ██ ███████     
    ██   ██ ██      ██   ██ ██      ██  ██  ██      ████  ████ ██    ██    ██   ██ ██          
    ██████  ██      ███████ ██      █████   ███████ ██ ████ ██ ██    ██    ███████ ███████     
    ██   ██ ██      ██   ██ ██      ██  ██       ██ ██  ██  ██ ██    ██    ██   ██      ██     
    ██████  ███████ ██   ██  ██████ ██   ██ ███████ ██      ██ ██    ██    ██   ██ ███████     
                                                                                            
                                                                                            
    ██████   ██████   ██████  ██   ██     ███████ ████████  ██████  ██████  ███████            
    ██   ██ ██    ██ ██    ██ ██  ██      ██         ██    ██    ██ ██   ██ ██                 
    ██████  ██    ██ ██    ██ █████       ███████    ██    ██    ██ ██████  █████              
    ██   ██ ██    ██ ██    ██ ██  ██           ██    ██    ██    ██ ██   ██ ██                 
    ██████   ██████   ██████  ██   ██     ███████    ██     ██████  ██   ██ ███████            
                                                                                            
                                                                                            
     ██████ ██   ██  █████  ████████ ██████   ██████  ████████                                 
    ██      ██   ██ ██   ██    ██    ██   ██ ██    ██    ██                                    
    ██      ███████ ███████    ██    ██████  ██    ██    ██                                    
    ██      ██   ██ ██   ██    ██    ██   ██ ██    ██    ██                                    
     ██████ ██   ██ ██   ██    ██    ██████   ██████     ██                                    
    """)
    print("Welcome to BlackSmith's Bookstore Chatbot!")

    # intent matching initialisations
    intents = readIntentsCsv()
    XtrainTf, count, tfidf = stemVectorWeight(intents, False, "XtrainTf.pickle", "count.pickle", "tfidf.pickle")
    invIdxIntents = generateInvertedIndex(count, XtrainTf, "invIdx.pickle")

    # QA search initialisations
    qa = readQaCsv()
    XtrainTfQa, countQa, tfidfQa = stemVectorWeight(qa, True, "XtrainTfQa.pickle", "countQa.pickle", "tfidfQa.pickle")
    invIdxQa = generateInvertedIndex(countQa, XtrainTfQa, "invIdxQa.pickle")

    while True:
        prompt = input("Please enter your prompt (QUIT to exit): ")
        if prompt.lower() == "quit":
            break
        intentResult = searchIntent(invIdxIntents, prompt, count, tfidf, XtrainTf, intents)
        if intentResult:
            intent = intents[intentResult[0]][1]
            if intent == "discover":
                discover()
            elif intent == "small":
                print(small(prompt))
            elif intent == "question":
                print(question(qa, prompt, countQa, tfidfQa, XtrainTfQa, invIdxQa))
            elif intent == "identity":
                print(identity(prompt))
            elif intent == "order":
                order(prompt)
            else:
                print("I'm not sure I understand. Could you try re-wording that?")
        else:
            print("I'm not sure I understand. Could you try re-wording that?")

        
    print("Goodbye!")

import datetime
import json
from nltk.corpus import stopwords
from nltk import download
download('stopwords', quiet=True)
from sklearn.feature_extraction.text import CountVectorizer

# ------ Pre-processing ------

import os
import csv
import pickle
def readIntentsCsv():
    '''
    Read in the CSV of example prompts labelled with respetive intents.
    As similarity based intent matching, resolve the user prompt to the 
    most similar entry in the corpus of prompt docs, resolve to respective intent label.
    The labels are one of:
    1. "small"
    2. "question"
    3. "identity"
    4. "discover"
    5. "order"
    6. "location"
    7. "check"
    8. "thank"
    '''
    # If we have the intents object saved to disk, just load and return that
    if os.path.exists('intents.pickle'):
        with open("intents.pickle", "rb") as f:
            return pickle.load(f)

    # If not saved to disk, re-load from csv
    intents = []
    with open('intents.csv', 'r', newline='') as f:
        r = csv.reader(f, delimiter=',')
        for row in r:
            # format: ["prompt","intent"]
            intents.append(row)
    
    # Save to disk for next time
    with open(f"intents.pickle", "wb") as f:
        pickle.dump(intents, f)

    return intents

# ------ Stemming, Vectorising, Weighting ------
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem.snowball import PorterStemmer
import re

p_stemmer = PorterStemmer()
def stemmed_words(doc):
    tokens = re.findall(r'\b\w+\b', doc.lower())
    return [p_stemmer.stem(token) for token in tokens]
def stemmed_filterStopwords_words(doc):
    tokens = re.findall(r'\b\w+\b', doc.lower())
    return [p_stemmer.stem(token) for token in tokens if token not in stopwords.words('english')]

def stemVectorWeight(intents: list, filterStopwords: bool, pName: str, pName1: str, pName2: str):
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
    for pair in intents: prompts.append(pair[0])

    # Initialise and run the count based vectoriser on the prompts, 
    # also using the stemmer.
    if filterStopwords:
        count_vect = CountVectorizer(tokenizer=stemmed_filterStopwords_words, lowercase=True)
    else:
        count_vect = CountVectorizer(tokenizer=stemmed_words, lowercase=True)
    X_train_counts = count_vect.fit_transform(prompts)

    # Term weighting: Term frequency - Inverse document frequency
    tf_transformer = TfidfTransformer(use_idf=True, sublinear_tf=True).fit(X_train_counts)
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
from collections import defaultdict

def createFloatDict():
    return defaultdict(float)

def generateInvertedIndex(count_vect, X_train_tf, pName):
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
    # Use dense conversion here; for typical coursework datasets this is fine.
    # If memory is a concern for larger collections, compute norms from sparse directly.
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

# ------ Search ------

'''
Use cosine similarity => cosine angle between the doc vector and the prompt vector.
'''
import json, re
from numpy import dot
from numpy.linalg import norm
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
    # Vectorise the query
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
        # TODO: make a clean mapping to translate arbitatry intent codes into more readable/natural wording.
        print(f"To confirm, you're asking about something {intents[similarity[0][0]][1]} related?")
        if confirmation():
            # If the user confirms, add their prompt + confirmed intent to the intents.csv dataset
            addIntentExample(query,intents[similarity[0][0]][1])
            return similarity[0]
        else:
            print("I'm sorry, I'm not sure what you mean then, please try re-wording your prompt or asking me 'what can you do' for specific examples.")

'''
Helper function for adding intent examples to the dataset.
'''
def addIntentExample(prompt, intent):
    with open('intents.csv', 'a', encoding='utf-8') as f:
        f.write(f'"{prompt}","{intent}"\n')

'''
Shared function for basic confirmation from the user.
Simplifies code in other handlers by providing a global abstracted function.
Returns simply True if they confirmed, False is they did not (assume False if no affirmation detected for safety.)
'''
def confirmation():
    answer = input("Please enter your prompt (QUIT to exit): ")
    if answer.lower() == "quit":
        exit()
    affirm = re.search(r"(?i)^\s*(?:yes|yep|yeah|y|sure|ok|okay|alright|affirmative|of course|definitely|certainly|sure thing|sounds good|roger)\b", answer) 
    if affirm:
        return True
    return False

'''
Simple handler for when the user thanks the chatbot
'''
def thank():
    if readName():
        print(f"You are very welcome, {readName()}! Glad to help.")
    else:
        print(f"You're welcome! Glad to help.")

'''
Small talk approach is based on 'ELIZA' from the course's recomended reading:
Pattern match prompt to identify keywords which can be saved and referred to in NLG template responses.
If no match, resort to generic response like "Why do you say that?" or "Tell me more".
Allow multiple matches from a single user prompt - gradually builds up a large response if the user's prompt was multi-faceted.
'''
def small(prompt):

    # List of regex patterns to match and capture specific keywords.
    # These keywords, depending on type, will be inserted into template responses. 

    # \b word boundary, ensures matches whole words
    # \s whitespace chars
    # \w word
    # (?:...) create non-capturing group

    patterns = [r"(?i)\b(?:feel|feeling)\s+(\w+)",
                r"(?i)\b(?:I|me)\b(?:\s+\w)?\s+feel\s+(.+?)\s+when\s+(.+)",
                r"(?i)\bwhen\s+(.+?)\s+(it\smakes\s)(?:i|me)\b(?:\s+\w)?\s+feel\s+(.+)",
                r"(?i)\b(?:how\sare\syou|how'?s\sit\sgoing|how'?s\sthings|what'?s\snew|how\shave\syou\sbeen)\b",
                r"(?i)\b(?:hi|hey|hello|howdy|greetings|good\s+(?:morning|afternoon|evening|day)|what'?s\sup|sup)\b"]

    emotion = re.search(patterns[0], prompt)
    emotionWithReason = re.search(patterns[1], prompt)
    emotionWithReason1 = re.search(patterns[2], prompt)
    howAre = re.search(patterns[3], prompt)
    greet = re.search(patterns[4], prompt)

    # Map for entity reference words, so they can be 'flipped' when the system uses them.
    # e.g: 'I feel sad when people laugh at me' -> 'why do you feel sad when people laugh at you'
    referenceMap = {
        'me': 'you',
        'i': 'you',
        'my': 'your',
    }

    # build up response gradually
    response = ""

    if greet:
        if readName():
            response += f"Hi, {readName()}. "
        else:
            print("Hello! ", end='')
            # re-use the identity function by putting a pseudo-prompt which will make the system ask for the user's name
            print(identity("what is my name"), end='')

    if emotionWithReason:
        reason = ""
        for word in re.findall(r"\w+",emotionWithReason.group(2)):
            if word in referenceMap.keys():
                reason += referenceMap[word] + " "
            else:
                reason += word + " "

        # [:-1] indexing on reason is simply to remove trailing whitespace before ?
        response += f"Why do you feel {emotionWithReason.group(1)} when {reason[:-1]}? "
    elif emotionWithReason1:
        reason = ""
        # instead of split(' '), this will also remove punctuation when tokenising
        for word in re.findall(r"\w+",emotionWithReason1.group(1)):
            if word in referenceMap.keys():
                reason += referenceMap[word] + " "
            else:
                reason += word + " "

        response += f"Why do you feel {emotionWithReason1.group(2)} when {reason[:-1]}? "
    
    elif emotion:
        response += f"Tell me more about why you feel {emotion.group(1)}. "

    if howAre:
        if readName():
            response += f"I'm doing well! How about you, {readName()}? "
        else:
            response += "I'm doing well! How about you? "
    
    if not response and not greet:
        response += "Tell me more."

    return response

'''
Handler for when intent is detected to be 'discover' which means the user is asking about what the chatbot can do.
Prints out a detailed explanation of capabilties to make the chatbot less of a 'black box'.
'''
def discover():
    print(
    "I can help with many things!\n",
    "Let me walk you through my features:\n"
    "   1. Book orders - I can help you order a book and set a delivery or pick-up time.\n",
    "   2. Book information - if you're not sure what you want to order, you can ask me about genres and authors and I'll recommend you a book.\n",
    "   3. Memory - I will remember your name and address you as such if you inform me of it!\n",
    "   4. Small talk - I can handle basic small talk if you'd like to chat.\n",
    "I hope this helps you converse with me effectively!")

def identity(prompt):
    # Use regular expression to determine specific intent (either they want to set the name or recall it)
    reIntents = [r"(?i)\b(?:my name is|call me|i am|i'm)\s+([A-Za-z][A-Za-z' -]*)",
                 r"(?i)\b(?:what\s+is\s+my\s+name|do\s+you\s+know\s+my\s+name|who\s+am\s+i)\b",
                 r"(?i)\b(?:forget\s+me|forget\s+my\s+(?:info|information|name|data)|delete\s+my\s+(?:info|information|name|data)|remove\s+my\s+data|erase\s+my\s+information)\b"]
    name = re.search(reIntents[0], prompt)
    recall = re.search(reIntents[1], prompt)
    forget = re.search(reIntents[2], prompt)
    if name:
        saveName(name.group(1))
        return f"Thanks for letting me know your name, {name.group(1)}. I'll remember that."
    elif recall:
        name = readName()
        if name:
            return f"I remember your name is {name}, of course!"
        else:
            # Need to a separate input here as they may respond with simply 'Joe' which the intents input loop may not be able to handle.
            print("I'm sorry, I don't think you've told me your name before. What is your name?")
            answer = input("Please enter your prompt (QUIT to exit): ")
            if answer.lower() == "quit":
                exit()
            uName = re.search(reIntents[0],answer)
            if not uName and len(answer.strip().split(' ')) == 1:
                saveName(answer.strip())
            else:
                saveName(uName.group(1))
            return f"Thanks for letting me know your name, {readName()}. I'll remember that."

    elif forget:
        # Confirm user's intention to remove their data from memory
        print("You want me to forget your information I've saved up until now?")
        if confirmation():
            resetSession()
            return "Okay I've forgotten any information I remembered about you."
        else:
            return "Okay, I won't forget your information."

        

    else:
        return "I'm sorry, I don't understand your indentity related request. Perhaps try re-wording your prompt."

def readName():
    # Read name from session JSON on disk
    if os.path.exists('session.json'):
        with open("session.json", "r") as f:
            session = json.load(f)
        return session['name']

def saveName(name):
    # Save the user details into a JSON on disk 
    # Check if a JSON exists already
    if os.path.exists('session.json'):
        # Exists already, so read it, update the name field, write back.
        with open("session.json", "r") as f:
            session = json.load(f)
        session['name'] = name
        with open("session.json", "w") as f:
            json.dump(session, f)
    else:
        session = {
            "name": name
        }
        with open("session.json", "w") as f:
            json.dump(session, f)

def resetSession():
    # Delete session JSON on disk
    os.remove("session.json")

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
        return "I'm sorry I'm not able to answer that with my current knowledge. \nMaybe try re-wording your question?"
    if qa[similarity[0][0]][1] == "Answer":
        return "I'm sorry I'm not able to answer that with my current knowledge. \nMaybe try re-wording your question?"
    return qa[similarity[0][0]][1]

def readQaCsv():
    # If we have the QA object saved to disk, just load and return that
    if os.path.exists('qa.pickle'):
        with open("qa.pickle", "rb") as f:
            return pickle.load(f)

    # If not saved to disk, re-load from csv
    qa = []
    with open('qa.csv', 'r', newline='') as f:
        r = csv.reader(f, delimiter=',')
        for row in r:
            # format: ["question","intent"]
            qa.append(row)
    
    # Save to disk for next time
    with open(f"qa.pickle", "wb") as f:
        pickle.dump(qa, f)

    return qa

# ------ Orders/Transactions ------

'''
Handles when the user wants a reccomendation for a book.
'''
def reccomend():
    # Get list of genres actually in the stock list
    genres = getGenres()
    


'''
Get all book JSONs of books in stock.json dataset.
'''
def getGenreBooks(genre: str) -> list:
    stock = getStockJSON()
    books = []
    for book in stock['stock']:
        if book['genre'] == genre:
            books.append(book)
    return books

'''
Get all genres of books in stock.json dataset.
'''
def getGenres() -> list[str]:
    stock = getStockJSON()
    genres = set()
    for book in stock['stock']:
        genres.add(book['genre'])
    return genres

'''
Handles when the user's intent is to check their existing orders.
'''
def check():
    pass

'''
Handles when a user's intent is to query about openining times / dates.
'''
def opening():
    pass

'''
Handles when a user's intent is to query about the bookstore's address.
'''
def address():
    pass

'''
Handles when a user's intent is to query about a location's facilities.
'''
def facilities():
    pass

'''
Use a slot filling approach.
Once this handler is called, enter a loop which follows the flow described below.
Flow:
1. scan the directed initial prompt for any already provided information.
2. for non-provided information, order of questions will be:
    - book 
    - quantity
    - pickup or delivery
    - for pickup:
        - ask location (check against actual locations)
        - ask pickup date and time (check against location opening hours and days)
        - confirm cost
    - for delivery:
        - ask address (check against some verification, plus provide a structure to use)
        - confirm cost (book + postage)
'''
def order(prompt: str):
    # Declare slots as None for now, any left as None after initial scan of prompt will be ask for
    book: str = None
    quantity: int = None
    pickup: bool = None
    address: str = None
    price: float = None
    name: str = None

    # Extract the book title they want to order.
    reTitleExtract = r"(?i)\b(?:order|buy|get|purchase|place)\b.*?\b([A-Za-z0-9'’:,&() ]{3,}?)\b(?=(?:\s+(?:for|from|to|at|in|pickup|delivery|delivered|store)\b|[.?!,;]|$))"
    title = re.search(reTitleExtract, prompt)
    # Extract the quantity they want to order.
    # Matches both numerical words and digits.
    reQuantityExtract = r"(?i)\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen| \
                        sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million)\b|\b(\d+)\b"
    numbers = re.search(reQuantityExtract, prompt)
    # Extract if it's for pickup for delivery.
    rePickupExtract = r"\b(pick-?up|drop-?off)\b"
    reDeliveryExtract = r"\b(delivery|home)\b"
    reDateExtractions = [
        # MM/DD 
        r'\b(\d{1,2}/\d{1,2})\b',                  
        # 10th of December
        r'\b(\d{1,2}(?:st|nd|rd|th)\s+of\s+\w+)\b',  
        # December 10th
        r'\b(\w+\s+\d{1,2}(?:st|nd|rd|th)?)\b',    
        # DD/MM/YY
        r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b'    
    ]
    pickupMatch = re.search(rePickupExtract, prompt)
    deliveryMatch = re.search(reDeliveryExtract, prompt)
    dateMatch = re.search(('|').join(reDateExtractions), prompt)

    # Check for book in stock.json 
    # Use Levenshtein-based fuzzy search to ensure detected book title can match 
    # to a specific title as per stock.json.
    exactTitle = None
    if title:
        exactTitle = fuzzySearchTitle(title.group(1))
    if exactTitle:
        # Extracted a title from the prompt and able to fuzzy find it in the stock dataset.
        # Set the book slot to the exact title.
        print(f"Okay! So you'd like to order: {exactTitle}?")
        if confirmation():
            book = exactTitle
    elif not exactTitle:
        # Extracted a title from the prompt, but unable to find it in the stock dataset.
        print("You'd like to place an order for which book? (please enter just the title)")
        answer = input("Please enter your prompt (QUIT to exit): ")
        if answer.lower() == "quit":
            exit()
        attempts = 0
        # Allow 3 re-entry attempts, if still no match, display list of titles in stock
        while True:
            exactTitle = fuzzySearchTitle(answer)
            if exactTitle: 
                print(f"Okay! So you'd like to order: {exactTitle}?")
                if confirmation():
                    book = exactTitle
                    break
                else:
                    answer = input("Please enter your prompt (QUIT to exit): ")
                    if answer.lower() == "quit":
                        exit()
            else:
                print("Sorry that didn't match any titles in our stock database, try again")
                answer = input("Please enter your prompt (QUIT to exit): ")
                if answer.lower() == "quit":
                    exit()
            attempts += 1
            if not (attempts > 3):
                print("Sorry, we don't stock that book.")
                break

    skip = False
    if book:
        if numbers:
            # Detected a numerical value in the user's prompt.
            # Confirm with user if this is the quantity they want to order.
            
            # Need to convert word for number into number.
            if numbers.group(1):
                quant = wordToInt(numbers.group(1).lower())
            else:
                quant = numbers.group(2)

            print(f"To confirm, you'd like to order {quant} copies?")
            if confirmation():
                quantity = quant
                skip = True
            else:
                print("Sorry for the misunderstanding, ", end='')
        if not skip:
            print("How many copies would you like?")
            answer = input("Please enter your prompt (QUIT to exit): ")
            if answer.lower() == "quit":
                exit()
            while True:
                numbers = re.search(reQuantityExtract, answer)
                if numbers:
                    if numbers.group(1):
                        quant = wordToInt(numbers.group(1))
                    else:
                        quant = numbers.group(2)
                    print(f"To confirm, you'd like to order {quant} copies?")
                    if confirmation():
                        quantity = quant
                        break
    
    if quantity and book:
        # Now have all the information needed to set the price for this order.
        price = getPrice(book) * float(quantity)
        if pickupMatch:
            # Keyword for pickup detected, but need to confirm.
            print("You would like to pick-up from one of our locations, right?")
            if confirmation():
                pickup = True
        elif deliveryMatch:
            # Keyword for delivery detected, but need to confirm.
            print("You would like this order to be for home delivery, right?")
            if confirmation():
                pickup = False
        else:
            # No decision is clear in the initial prompt, ask directly
            print("For pickup or delivery? (type pickup/delivery)")
            answer = input("Please enter your prompt (QUIT to exit): ")
            if answer.lower() == "quit":
                exit()
            elif answer.find("delivery") != -1:
                pickup = False
            else:
                pickup = True
        # Based on pickup boolean, get home address details or get store location.
        if pickup:
            attempts = 0
            while attempts < 4:
                print("Which BlackSmith™'s store location would you like to pick-up your order from? (type 'list' to get a list of all locations)")
                answer = input("Please enter your prompt (QUIT to exit): ")
                if answer.lower() == "quit":
                    exit()
                elif answer.lower().find("list") != -1:
                    # Help the user discover what locations exist.
                    print("Our bookstores can be found in the following locations:")
                    for location in getAllLocations():
                        print(f"Location: {location[0]}, Address: {location[1]}")
                else:
                    # Search for the location they specified.
                    # After 3 fails to recognise a location name, print out the list of locations even if the user didn't ask.
                    location = extractLocation(answer)
                    if location:
                        print(f'Okay! I have set your order to be picked up from the BlackSmith store in {location}!')
                        address = location
                        break
                    else:
                        attempts += 1
                        if attempts == 2:
                            # Help the user discover what locations exist.
                            # Elaborate on the expected format of the response from the user.
                            print("If you are entering the address and I'm failing to recognise it, I am just looking for the general location, e.g: 'London'.")
                            print("For your reference, our bookstores can be found in the following locations:")
                            for location in getAllLocations():
                                print(f"Location: {location[0]}, Address: {location[1]}")
                        if attempts > 3:
                            print("I'm sorry I'm unable to understand which location you'd like to pickup from.\n" \
                            "Would you like to cancel this order? If not, I'll keep trying to understand which location you'd prefer.")
                            if confirmation():
                                print("Okay! I'll keep trying!")
                                attempts = 0
                            else:
                                print("Okay, I'm sorry I failed to understand your desired location. \nI have cancelled this order.")
                                # TODO: Feedback mechanism - ask for user feedback (rating + open feedback) and store for developer use.
                                break
            if address:
                # If the user successfully selected a pickup location, 
                # Need to select a pick-up date and timeslot.
                # Need to prevent selecting a date when the specific location is closed, 
                # or a time when the location is closed.
                # TODO get date and time implementation
                date = getPickupDate()
                time = getPickupTime()

        else:
            pass
        
    if address:
        # Check if we know the user's name, if we do not, ask for the order and save for later too.
        # If we do, just use that for the order name.
        nameSaved = readName()
        if nameSaved:
            name = nameSaved
        else:
            # Force the identity function to ask the user for their name and save it.
            identity("What is my name")
            name = readName()
            print(f"I've set the name for your order as {name}")
    
    if book and pickup and quantity and price and address and name:
        storeOrder(book, getISBN(book), quantity, pickup, address, None, None, price, name)

def getISBN(book: str) -> str:
    pass

def getPickupDate():
    pass
    
def getPickupTime():
    pass

'''
Returns the float price for a book based on it's title.
If name is not found, returns -1
'''
def getPrice(title: str) -> float:
    titleNorm = title.lower().strip()
    for book in getStockJSON()['stock']:
        if book['name'].lower().strip() == titleNorm:
            return book['price']
    return -1

'''
Attempts to extract a location from the prompt which matches one in the locations.json dataset.
If fail to extract or match, returns the empty string.
'''
def extractLocation(prompt: str) -> str:
    locations = getAllLocations()
    # Use a map to just have a 1-D list of locations, removing address.
    locations = list(map(lambda x: x[0].lower(), locations))

    # Tokenise into words and lowercase.
    tokens = re.findall(r'\b\w+\b', prompt.lower())
    stopWords = stopwords.words('english')
    # Remove all stopwords from the tokenized prompt.
    filteredTokens = [t for t in tokens if t not in stopWords]

    if not filteredTokens:
        # Means empty prompt or all stop words
        # As a fallback, attempt to directly match locations in the raw, unfiltered prompt
        promptLower = prompt.lower()
        for loc in locations:
            if re.search(r'\b' + re.escape(loc) + r'\b', promptLower):
                return loc
        return ''
    
    # Maximum number of words in any stored location name
    maxLenLoc = max(len(loc) for loc in locations)
    # Try the longest n-grams first for multiple word location names 
    for n in range(maxLenLoc, 0, -1):
        for i in range(len(filteredTokens) - n + 1):
            gram = ' '.join(filteredTokens[i:i+n])
            if gram in locations:
                return gram
    return ''

'''
Returns a list of all location names and addresses in the locations.json dataset.
'''
def getAllLocations() -> list[list[str]]:
    locations = getLocationsJSON()
    allLocs = []
    for location in locations['locations']:
        loc = []
        loc.append(location['name'])
        loc.append(location['address'])
        allLocs.append(loc)
    return allLocs

'''
Returns the JSON data for bookstore locations.
'''
def getLocationsJSON():
    with open('locations.json', 'r') as f:
        data = json.load(f)
    return data

'''
Takes day, month, year of a date.
Returns the integer Unix epoch timestamp.
For simple storage and standardised data representation.
'''
def getUnixEpochTimestamp(dd: int,mm: int,yyyy: int)-> int:
    date_obj = datetime.datetime(yyyy, mm, dd)
    return int(date_obj.timestamp())

'''
Convert Unix epoch timestamp into datetime object for easy printing when user queries their order.
'''
def getDateFromUnix(timestamp: int):
    date_obj = datetime.datetime.fromtimestamp(timestamp)
    return date_obj

'''
Resolves words for numbers a user may enter into their integer form for easy processing.
'''
def wordToInt(word) -> int:
    wordToInt = {
        'one': 1,
        'two': 2,
        'three': 3,
        'four': 4,
        'five': 5,
        'six': 6,
        'seven': 7,
        'eight': 8,
        'nine': 9,
        'ten': 10,
        'eleven': 11,
        'twelve': 12,
        'thirteen': 13,
        'fourteen': 14,
        'fifteen': 15,
        'sixteen': 16,
        'seventeen': 17,
        'eighteen': 18,
        'nineteen': 19,
        'twenty': 20,
        'thirty': 30,
        'forty': 40,
        'fifty': 50,
        'sixty': 60, 
        'seventy': 70,
        'eighty': 80,
        'ninety': 90,
        'hundred': 100,
        'thousand': 1000,
        'million': 1000000
    }
    if word in wordToInt:
        return wordToInt[word]
    else:
        return -1

'''
Perform a fuzzy search for title using Levenshtein distance.
Returns the true title per stock.json, or empty string if not found.
'''
def fuzzySearchTitle(title: str) -> str:
    titleNorm = title.lower().strip()
    levenshteinDistances = []
    for book in getStockJSON()['stock']:
        bookNorm = book['name'].lower().strip()
        # Calculate distance between user title and the stored book title
        levenshteinDistances.append((book['name'],levenshteinDistance(bookNorm, titleNorm, len(bookNorm), len(titleNorm))))
    # Sort ascending, first index will be the most similar / least distance
    levenshteinDistances.sort(key=lambda entry: entry[1])
    # Tolerance is a levenshtein distance of 1/3rd of the user's desired title
    if levenshteinDistances and levenshteinDistances[0][1] < len(title)//3:
        return levenshteinDistances[0][0]
    else:
        return ""

def getStockJSON():
    with open('stock.json', 'r') as f:
        data = json.load(f)
    return data

def getOrdersJSON():
    if os.path.exists('orders.json'):
        with open('orders.json', 'r') as f:
            data = json.load(f)
        return data
    return None

def storeOrder(title, isbn, quantity, pickup, address, date, time, cost, name):
    orders = getOrdersJSON()
    order = {
        "title": title,
        "ISBN": isbn,
        "quantity": quantity,
        "pickup": pickup,
        "address": address,
        "date": date,
        "time": time,
        "cost": cost,
        "name": name
    }
    if orders:
        orders['orders'].append(order)
    else:
        orders = {
                "orders": [order]
        }
    ordersStr = json.dumps(orders, indent=4)
    with open('orders.json', 'w') as f:
        f.write(ordersStr)

# TODO add genre to stocks to allow genre based recomendations.

'''
Returns the Levenshtein distance which can be used to implement a fuzzy search.
(To be applied to matching a user's desired book title, even if slightly off, to the name in the stock.json dataset.)

Core concept of Levenshtein distance:
- Measures difference between 2 strings,
- by counting how many single character edits are needed to reach one from the other.
Operations include:
- Insertion (add char)
- Deletion (remove char)
- Substitution (replace a char with another)

Below implementation uses a recursive approach:

'm' and 'n' are the current lengths being considered for each string respectively.
Initialised to len(a) and len(b).

If m is 0, the distance is the remaining chars in the other string (as we'd need n insertions.)
If n is 0, the distance is the number of remaining chars in the other string (as we'd need m deletions.)

If last characters match (a[m-1] == b[n-1]) then no need to edit that char, recurse on prefixes.

If last characters differ, recur with all 3 potential operations to match the chars.
'''
def levenshteinDistance(a: str, b: str, m: int, n: int):
    # Using a dynamic programming approach for efficiency.
    # O(m*n) time complexity.
    # How it works is by building up a 2D array where each entry dp[i][j] represents the Levenshtein distance
    # between the first i characters of string a and the first j characters of string b.
    # this means that we can build up the solution for larger substrings based on the solutions for smaller substrings.

    # Create a 2D array to store distances between prefixes of the strings.
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    # Fill dp array 
    for i in range(1, m + 1):
        # Current character of a
        ai = a[i - 1]
        # Iterate through each character of b
        for j in range(1, n + 1):
            bj = b[j - 1]
            # Cost of substitution
            cost = 0 if ai == bj else 1
            # Compute minimum cost possible between: deletion, insertion, substitution
            dp[i][j] = min(
                # deletion
                dp[i - 1][j] + 1,      
                # insertion
                dp[i][j - 1] + 1,      
                # substitution
                dp[i - 1][j - 1] + cost  
            )
    # The bottom right of the 2D DP array contains the Levenshtein distance built up
    return dp[m][n]

if __name__ == "__main__":
    main()
