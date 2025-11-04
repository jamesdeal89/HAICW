
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
5. Small talk. [ ]
'''

'''
INITIAL IDEAS:
Book ordering system.
Transaction: ordering books, scheduling pickup/delivery.
Q&A: book recomendations, location opening times.
'''

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
'''

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
    invIdx = generateInvertedIndex(count, XtrainTf, "invIdx.pickle")

    # QA search initialisations
    qa = readQaCsv()
    XtrainTfQa, countQa, tfidfQa = stemVectorWeight(qa, True, "XtrainTfQa.pickle", "countQa.pickle", "tfidfQa.pickle")
    invIdx = generateInvertedIndex(countQa, XtrainTfQa, "invIdxQa.pickle")

    while True:
        prompt = input("Please enter your prompt (QUIT to exit): ")
        if prompt.lower() == "quit":
            break
        intent = intents[searchIntent(invIdx, prompt, count, tfidf, XtrainTf, intents)[0]][1]
        if intent == "discover":
            discover()
        elif intent == "small":
            small(prompt)
        elif intent == "question":
            print(question(qa, prompt, countQa, tfidfQa, XtrainTfQa))
        elif intent == "identity":
            print(identity(prompt))
        else:
            print("I'm not sure I understand. Could you try re-wording that?")
        
    print("Goodbye!")

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
    # Save to disk for next time
    with open(pName, "wb") as f:
        pickle.dump(inverted_index, f)
    return inverted_index

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
    # Convert to dense arrays for consistent dimensions
    qVec = qTfidf.toarray().flatten()
    dVecs = X_train_tf.toarray()

    similarity = []
    # Calculate cosine similarity to each doc's vector
    for docId in range(len(intents)):
        dVec = dVecs[docId]
        similarity.append((docId,getCosineSimilarity(qVec, dVec)))
    # Return the most likely intent
    '''
    Intent can be one of:
        1. "small"
        2. "question"
        3. "identity"
        4. "discover"
    '''
    similarity.sort(key = lambda x:x[1], reverse=True)
    return similarity[0]


# TODO: small talk 

def small(prompt):
    print("Small talk coming soon!")

def discover():
    print(
    "I can help with many things!\n",
    "Let me walk you through my features:\n"
    "   1. Book orders - I can help you order a book and set a delivery or pick-up time.\n",
    "   2. Book information - if you're not sure what you want to order, you can ask me about genres and authors and I'll recommend you a book.\n",
    "   3. Memory - I will remeber your name and address you as such if you inform me of it!\n",
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
            return "I'm sorry, I don't think you've told me your name before. What is your name?"
    elif forget:
        # Confirm user's intention to remove their data from memory
        print("You want me to forget your information I've saved up until now?")
        answer = input("Please enter your prompt (QUIT to exit): ")
        if answer.lower() == "quit":
            exit()
        affirm = re.search(r"(?i)^\s*(?:yes|yep|yeah|y|sure|ok|okay|alright|affirmative|of course|definitely|certainly|sure thing|sounds good|roger)\b", answer) 
        if affirm:
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

def question(qa, question, vectoriser, tfidf, X_train_tf):
    # Vectorise the query 
    qCounts = vectoriser.transform([question])
    qTfidf = tfidf.transform(qCounts)
    # Convert to dense arrays for consistent dimensions
    qVec = qTfidf.toarray().flatten()
    dVecs = X_train_tf.toarray()

    similarity = []
    # Calculate cosine similarity to each doc's vector
    for docId in range(len(qa)):
        dVec = dVecs[docId]
        similarity.append((docId,getCosineSimilarity(qVec, dVec)))
    # Return the most likely answer
    similarity.sort(key = lambda x:x[1], reverse=True)
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
            qa.append(row[1:3])
    
    # Save to disk for next time
    with open(f"qa.pickle", "wb") as f:
        pickle.dump(qa, f)

    return qa

if __name__ == "__main__":
    main()
