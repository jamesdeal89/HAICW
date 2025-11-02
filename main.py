
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
1. Intent matching. [ ]
2. Identity matching. [ ]
3. Transactions. [ ]
4. Information retrieval & Question answering. [ ]
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

if __name__ == "__main__":
    main()

# TODO: similarity-based intent matching
from nltk.corpus import stopwords
# nltk.download('stopwords', quiet=True)
from sklearn.feature_extraction.text import CountVectorizer

# ------ Pre-processing ------

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

    intents = []
    import csv
    with open('intents.csv', 'r', newline='') as f:
        r = csv.reader(f, delimiter=',')
        for row in r:
            # format: ["prompt","intent"]
            intents.append(row)
    return intents
    # TODO: pickle intents object so it doesn't re-load every run

# ------ Stemming, Vectorising, Weighting ------
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem.snowball import PorterStemmer
def stemVectorWeight(intents):
    # Initialise a stemmer to use with the vectoriser
    p_stemmer = PorterStemmer()
    analyser = CountVectorizer.build_analyzer()
    def stemmed_words(doc):
        return (p_stemmer.stem(w) for w in analyser(doc))

    # Get just the prompts to vectorise, but maintain indexing to resolve to labels.
    prompts = []
    for pair in intents: prompts.append(pair[0])

    # Initialise and run the count based vectoriser on the prompts, 
    # filtering out stop-words, and using the stemmer.
    count_vect = CountVectorizer(stop_words=stopwords.words('english'), analyzer=stemmed_words)
    X_train_counts = count_vect.fit_transform(prompts)

    # Term weighting: Term frequency - Inverse document frequency
    tf_transformer = TfidfTransformer(use_idf=True, sublinear_tf=True).fit(X_train_counts)
    return (tf_transformer.transform(X_train_counts), count_vect)

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
def generateInvertedIndex(count_vect, X_train_tf):
    # Format: {term: {docId: tfidfScore}}
    inverted_index = defaultdict(lambda: defaultdict(int))

    feature_names = count_vect.get_feature_names_out()
    # Convert to co-ordinate format to make iteration more efficient.
    tfid_matrix = X_train_tf.tocoo()
    for docId, termId, score in zip(tfid_matrix.row, tfid_matrix.col, tfid_matrix.data):
        term = feature_names[termId]
        inverted_index[term][docId] = score
    return inverted_index

# 'Postings' => list of docs that contain each term, alongside the term's importance in those docs.
# inverted_index resolves a term in the vocab to it's respective postings.

# ------ Search ------

'''
Use cosine similarity => cosine angle between the doc vector and the prompt vector.
'''
from numpy import dot
from numpy.linalg import norm
def getCosineSimilarity(query_doc, doc):
    return dot(query_doc, doc) / (norm(query_doc) * norm(doc))


# TODO: small talk 

# TODO: discoverability

# TODO: user personalisation / memory

# TODO: Q&A with stocks & bonds dataset provided for checkpoint



