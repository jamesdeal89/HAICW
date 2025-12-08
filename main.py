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
3. Transactions. [x]
4. Information retrieval & Question answering. [x]
5. Small talk. [x]
'''

'''
CORE IDEA:
Book ordering, reccomendation and information system.
Transaction: ordering books, scheduling pickup/delivery.
Q&A: book recomendations, location opening times.
'''

'''
NOTE on Q&A dataset:
Static dataset was partially genAI generated to bulk out examples. 
NONE of the code, processing, etc are AI generated.
NOTE on intents dataset:
Based on boostrapped manual examples, expanded with genAI.
Bulked out using confirmation:
    - if the user confirms a low confidence intent match, 
        - add their prompt and the confirmed intent to dataset.
'''

# Prevent warnings which may occur from pickling, sometimes versions may mistmatch, but has no notable effect.
import warnings
warnings.filterwarnings('ignore')

from preprocessing import readIntentsCsv, stemVectorWeight, generateInvertedIndex, readQaCsv, readBookDescriptions, clearOldPickles
from search import searchIntent, question
from handlers import discover, small, identity, thank, reccomend, check, opening, address, facilities, locations, stockCheck
from orders import order
from context import resolveEllipsis
import os

def main() -> None:
    # Clear old pickle caches if they're older than 1 day
    clearOldPickles()
    
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

    # Intent matching initialisations
    intents = readIntentsCsv()
    XtrainTf, count, tfidf = stemVectorWeight(intents, False, "XtrainTf.pickle", "count.pickle", "tfidf.pickle")
    invIdxIntents = generateInvertedIndex(count, XtrainTf, "invIdx.pickle")

    # QA search initialisations
    qa = readQaCsv()
    XtrainTfQa, countQa, tfidfQa = stemVectorWeight(qa, True, "XtrainTfQa.pickle", "countQa.pickle", "tfidfQa.pickle")
    invIdxQa = generateInvertedIndex(countQa, XtrainTfQa, "invIdxQa.pickle")

    # Book description search initialisations
    bookDesc = readBookDescriptions()
    XtrainTfBookDesc, countBookDesc, tfidfBookDesc = stemVectorWeight(bookDesc, False, "XtrainTfBookDesc.pickle", "countBookDesc.pickle", "tfidfBookDesc.pickle")
    invIdxBookDesc = generateInvertedIndex(countBookDesc, XtrainTfBookDesc, "invIdxBookDesc.pickle")

    # First use 
    if not os.path.exists('session.json'):
        # Asks user for their name.
        print(small("hello"))
    
    # Main conversational loop and respective intent handler calls.
    while True:
        prompt = input("Please enter your prompt (QUIT to exit): ")
        if prompt.lower() == "quit":
            break
        resolvedPrompt = resolveEllipsis(prompt)
        intentResult = searchIntent(invIdxIntents, resolvedPrompt, count, tfidf, intents)
        if intentResult:
            intent = intents[intentResult[0]][1]
            if intent == "discover":
                discover()
            elif intent == "small":
                print(small(resolvedPrompt))
            elif intent == "question":
                print(question(qa, resolvedPrompt, countQa, tfidfQa, invIdxQa))
            elif intent == "identity":
                print(identity(resolvedPrompt))
            elif intent == "thank":
                print(thank())
            elif intent == "reccomend":
                reccomend(resolvedPrompt, bookDesc, countBookDesc, tfidfBookDesc, invIdxBookDesc)
            elif intent == "check":
                check()
            elif intent == "order":
                order(resolvedPrompt, intents, count, tfidf, invIdxIntents, 
                      qa, countQa, tfidfQa, invIdxQa,
                      bookDesc, countBookDesc, tfidfBookDesc, invIdxBookDesc)
            elif intent == "opening":
                opening(resolvedPrompt)
            elif intent == "address":
                address(resolvedPrompt)
            elif intent == "facilities":
                facilities(resolvedPrompt)
            elif intent == "locations":
                locations()
            elif intent == "stockCheck":
                stockCheck(resolvedPrompt)
            else:
                print("I'm not sure I understand. Could you try re-wording that?")
        else:
            print("I'm not sure I understand. Could you try re-wording that?")

        
    print("Goodbye!")

if __name__ == "__main__":
    main()
